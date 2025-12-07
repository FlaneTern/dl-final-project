from my_utils.seed_utils import set_seed

set_seed(42)   # call once at the very top, before dataloaders/models


# %% [markdown]
# Common setup: paths for models and results

from pathlib import Path
import json
import shutil
import time

MODELS_DIR = Path("trained_models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% 
# ======================================
# 1. Faster R-CNN: train + COCO mAP + val loss
# ======================================
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from my_utils.coco_utils import make_coco_loaders
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_frcnn_model(num_classes=91):  # 80 classes + background + some extra
    # pretrained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"  # new API
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



def get_coco_api_from_loader(loader):
    """
    Robustly get the underlying pycocotools COCO object from a DataLoader.
    """
    ds = loader.dataset
    for _ in range(10):
        if hasattr(ds, "coco"):
            return ds.coco
        if hasattr(ds, "dataset"):
            ds = ds.dataset
        else:
            break
    raise AttributeError(
        "Could not find 'coco' attribute in dataset. "
        "Please check make_coco_loaders implementation."
    )


def get_img_ids_for_loader(loader):
    """
    Build a list img_ids such that:
        img_ids[i] = COCO image_id corresponding to loader.dataset[i]

    Handles Subset and simple wrappers. Assumes val_loader has shuffle=False.
    """
    ds = loader.dataset
    indices = None  # indices in the base dataset

    while True:
        if isinstance(ds, Subset):
            if indices is None:
                indices = list(ds.indices)
            else:
                indices = [indices[i] for i in ds.indices]
            ds = ds.dataset
            continue

        if not hasattr(ds, "coco") and hasattr(ds, "dataset"):
            ds = ds.dataset
            continue

        break

    if not hasattr(ds, "coco"):
        raise RuntimeError(
            "Could not find a base COCO dataset with a 'coco' attribute under loader.dataset"
        )

    base_ds = ds
    coco = base_ds.coco

    if indices is None:
        indices = list(range(len(base_ds)))

    if hasattr(base_ds, "ids"):
        base_img_ids = list(base_ds.ids)
    else:
        base_img_ids = list(sorted(coco.getImgIds()))

    loader_img_ids = [int(base_img_ids[i]) for i in indices]
    return loader_img_ids


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="Train FRCNN"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
    return running_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    """
    Compute validation loss for Faster R-CNN.

    NOTE: torchvision detection models only return a loss dict when
    model.training == True, so we temporarily switch to train() here,
    but keep gradients disabled with torch.no_grad().
    """
    # remember previous mode
    was_training = model.training

    # must be train() to get loss dict from torchvision detection models
    model.train()

    running_loss = 0.0

    for images, targets in tqdm(loader, desc="Val FRCNN (loss)"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)   # this is now a dict
        losses = sum(loss for loss in loss_dict.values())

        running_loss += losses.item()

    # restore previous mode
    if was_training:
        model.train()
    else:
        model.eval()

    return running_loss / len(loader)


@torch.no_grad()
def evaluate_coco_mAP(model, loader, device):
    """
    Run COCO-style evaluation on of val2017.
    Returns dict with AP, AP50, AP75, APs, APm, APl.
    """
    model.eval()
    coco = get_coco_api_from_loader(loader)
    coco_img_ids_all = set(coco.getImgIds())

    loader_img_ids = get_img_ids_for_loader(loader)
    assert len(loader_img_ids) == len(loader.dataset), \
        "Length mismatch between loader_img_ids and loader.dataset"

    results = []
    global_idx = 0

    for images, _targets in tqdm(loader, desc="Eval FRCNN (COCO mAP)"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        batch_size = len(outputs)
        batch_img_ids = loader_img_ids[global_idx: global_idx + batch_size]
        global_idx += batch_size

        for img_id, output in zip(batch_img_ids, outputs):
            if img_id not in coco_img_ids_all:
                continue

            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            if boxes.numel() == 0:
                continue

            # xyxy -> xywh
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

            for box, score, label in zip(boxes_xywh, scores, labels):
                results.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(label),  # COCO cat_ids
                        "bbox": box.tolist(),
                        "score": float(score),
                    }
                )

    if not results:
        print("No detections to evaluate.")
        return None

    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")

    eval_img_ids = sorted({r["image_id"] for r in results})
    coco_eval.params.imgIds = eval_img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP":   float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "APs":  float(coco_eval.stats[3]),
        "APm":  float(coco_eval.stats[4]),
        "APl":  float(coco_eval.stats[5]),
    }
    return metrics


# ------------ main FRCNN script ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader = make_coco_loaders(
    root="/mnt/ssd2/santana-coco/data/coco",
    batch_size=2,
    num_workers=4,
    train_limit=5000,   # None for full train2017 per proposal
    val_limit=1000,     # None for full val2017 per proposal
)

num_classes = 91  # standard COCO setting (incl. background)
model = get_frcnn_model(num_classes=num_classes).to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
best_ap = 0.0

frcnn_history = []

# Evaluate pre-trained FRCNN before fine-tuning
print("Evaluating FRCNN before training...")
metrics0 = evaluate_coco_mAP(model, val_loader, device)
print("Initial FRCNN metrics:", metrics0)

# match DETR history structure: epoch 0, no train/val loss yet
entry0 = {"epoch": 0, "train_loss": None, "val_loss": None}
if metrics0 is not None:
    entry0.update(metrics0)
frcnn_history.append(entry0)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, device)
    scheduler.step()

    metrics = evaluate_coco_mAP(model, val_loader, device)
    if metrics is not None:
        ap = metrics["AP"]
        print(
            f"[FRCNN] Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"AP={ap:.4f} | AP50={metrics['AP50']:.4f} | "
            f"AP75={metrics['AP75']:.4f} | APs={metrics['APs']:.4f} | "
            f"APm={metrics['APm']:.4f} | APl={metrics['APl']:.4f}"
        )

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history_entry.update(metrics)
        frcnn_history.append(history_entry)

        if ap > best_ap:
            best_ap = ap
            torch.save(
                model.state_dict(),
                MODELS_DIR / "frcnn_best.pth",
            )
            print(f"  -> New best FRCNN model saved (AP={ap:.4f})")
    else:
        print(
            f"[FRCNN] Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"no detections on val set"
        )
        frcnn_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    torch.save(
        model.state_dict(),
        MODELS_DIR / f"frcnn_epoch{epoch + 1}.pth",
    )


@torch.no_grad()
def measure_fps_detector(model, loader, device, warmup=20, max_batches=100):
    model.eval()
    num_images = 0

    # 1) warmup
    it = iter(loader)
    for _ in range(warmup):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        images = [img.to(device) for img in images]
        _ = model(images)

    # 2) timed loop
    it = iter(loader)
    start = time.perf_counter()
    for _ in range(max_batches):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        images = [img.to(device) for img in images]
        _ = model(images)
        num_images += len(images)
    end = time.perf_counter()

    elapsed = end - start
    fps = num_images / elapsed if elapsed > 0 else 0.0
    return fps

fps_frcnn = [measure_fps_detector(model, val_loader, device)]
print("FRCNN FPS (val subset):", fps_frcnn)
with open(RESULTS_DIR / "frcnn_fps.json", "w") as f:
    json.dump(fps_frcnn, f, indent=2)


# Save FRCNN history for later plotting
with open(RESULTS_DIR / "frcnn_history.json", "w") as f:
    json.dump(frcnn_history, f, indent=2)
print("Saved FRCNN history to", RESULTS_DIR / "frcnn_history.json")
