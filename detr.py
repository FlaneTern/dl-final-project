from src.utils.seed_utils import set_seed

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
# 3. HF DETR: train + val loss + COCO mAP
# ======================================
import torch
from torch.optim import AdamW
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import Subset  # used in get_img_ids_for_loader

from src.utils.coco_utils import make_coco_loaders


def get_img_ids_for_loader(loader):
    """
    Build a list img_ids such that:
        img_ids[i] = COCO image_id corresponding to loader.dataset[i]

    This walks through possible wrappers (Subset, custom DatasetWrapper, etc.)
    and recovers the base COCO dataset's id list.
    Assumes val_loader is created with shuffle=False.
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


# -------------------------
# Helper: xyxy → HF targets
# -------------------------
def build_hf_targets(targets):
    """
    Convert a batch of targets from your format:
        {
            "boxes": Tensor[num_boxes, 4] in xyxy,
            "labels": Tensor[num_boxes],
            (optionally "image_id", "area", "iscrowd")
        }
    into HF/COCO-style annotations.
    """
    hf_targets = []

    for t in targets:
        boxes = t["boxes"]  # (N, 4), xyxy
        labels = t["labels"]

        if boxes.numel() == 0:
            annotations = []
        else:
            xywh = boxes.clone()
            xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
            xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

            annotations = []
            for box, label in zip(xywh, labels):
                bbox = box.tolist()
                category_id = int(label.item() if torch.is_tensor(label) else label)
                ann = {
                    "bbox": bbox,
                    "category_id": category_id,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
                annotations.append(ann)

        if "image_id" in t:
            if torch.is_tensor(t["image_id"]):
                image_id = int(t["image_id"].item())
            else:
                image_id = int(t["image_id"])
        else:
            image_id = 0

        hf_targets.append({
            "image_id": image_id,
            "annotations": annotations,
        })

    return hf_targets


# -------------------------
# Training loop for HF DETR
# -------------------------
def train_one_epoch_detr_hf(model, processor, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc="Train HF-DETR"):
        hf_targets = build_hf_targets(targets)

        encoding = processor(
            images=list(images),
            annotations=hf_targets,
            return_tensors="pt",
            do_rescale=False,      # <<< IMPORTANT: our images are already in [0,1]
        )

        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding["pixel_mask"].to(device)   # <<< use the mask
        labels = [
            {k: v.to(device) for k, v in target.items()}
            for target in encoding["labels"]
        ]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

@torch.no_grad()
def validate_one_epoch_detr_hf(model, processor, loader, device):
    model.eval()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc="Val HF-DETR (loss)"):
        hf_targets = build_hf_targets(targets)

        encoding = processor(
            images=list(images),
            annotations=hf_targets,
            return_tensors="pt",
            do_rescale=False,      # <<< same fix
        )

        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding["pixel_mask"].to(device)
        labels = [
            {k: v.to(device) for k, v in target.items()}
            for target in encoding["labels"]
        ]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        loss = outputs.loss

        running_loss += loss.item()

    return running_loss / len(loader)


# -------------------------
# COCO mAP evaluation for HF DETR
# -------------------------
@torch.no_grad()
def evaluate_coco_mAP_detr(model, processor, loader, device):
    model.eval()

    coco = get_coco_api_from_loader(loader)
    coco_img_ids_all = set(coco.getImgIds())

    loader_img_ids = get_img_ids_for_loader(loader)
    assert len(loader_img_ids) == len(loader.dataset)

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    label_idx_to_cat_id = {}
    for idx, name in id2label.items():
        cat_ids = coco.getCatIds(catNms=[name])
        if len(cat_ids) > 0:
            label_idx_to_cat_id[idx] = cat_ids[0]

    if not label_idx_to_cat_id:
        print("WARNING: could not map any DETR labels to COCO category ids.")
        return None

    results = []
    global_idx = 0

    for images, _targets in tqdm(loader, desc="Eval HF-DETR (COCO mAP)"):
        images = list(images)

        target_sizes = []
        for img in images:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            target_sizes.append([h, w])

        encoding = processor(
            images=images,
            return_tensors="pt",
            do_rescale=False,      # <<< same fix
        )

        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding["pixel_mask"].to(device)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        processed_outputs = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor(target_sizes, device=device),
            threshold=0.0,
        )

        batch_size = len(processed_outputs)
        batch_img_ids = loader_img_ids[global_idx: global_idx + batch_size]
        global_idx += batch_size

        for img_id, pred in zip(batch_img_ids, processed_outputs):
            if img_id not in coco_img_ids_all:
                continue

            boxes = pred["boxes"].detach().cpu()
            scores = pred["scores"].detach().cpu()
            labels = pred["labels"].detach().cpu()

            if boxes.numel() == 0:
                continue

            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

            for box, score, label in zip(boxes_xywh, scores, labels):
                label_idx = int(label)
                if label_idx not in label_idx_to_cat_id:
                    continue

                cat_id = int(label_idx_to_cat_id[label_idx])

                results.append(
                    {
                        "image_id": int(img_id),
                        "category_id": cat_id,
                        "bbox": box.tolist(),
                        "score": float(score),
                    }
                )

    # ... rest of the function unchanged ...

    if not results:
        print("No detections to evaluate (results list is empty).")
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


# ------------ main DETR script ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader = make_coco_loaders(
    root="/mnt/ssd2/santana-coco/data/coco",
    batch_size=2,
    num_workers=4,
    train_limit=5000,
    val_limit=1000,
)

processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50",
    revision="no_timm",
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    revision="no_timm",
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 10
best_ap = 0.0
detr_history = []

# Optional: evaluate DETR before fine-tuning
print("Evaluating HF-DETR before training...")
metrics0 = evaluate_coco_mAP_detr(
    model=model,
    processor=processor,
    loader=val_loader,
    device=device,
)
print("Initial HF-DETR metrics:", metrics0)

entry0 = {"epoch": 0, "train_loss": None, "val_loss": None}
if metrics0 is not None:
    entry0.update(metrics0)
detr_history.append(entry0)

for epoch in range(num_epochs):
    train_loss = train_one_epoch_detr_hf(
        model=model,
        processor=processor,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
    )

    val_loss = validate_one_epoch_detr_hf(
        model=model,
        processor=processor,
        loader=val_loader,
        device=device,
    )

    metrics = evaluate_coco_mAP_detr(
        model=model,
        processor=processor,
        loader=val_loader,
        device=device,
    )

    if metrics is not None:
        ap = metrics["AP"]
        print(
            f"[HF-DETR] Epoch {epoch + 1}/{num_epochs} | "
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
        detr_history.append(history_entry)

        if ap > best_ap:
            best_ap = ap
            torch.save(
                model.state_dict(),
                MODELS_DIR / "detr_hf_best.pth",
            )
            print(f"  -> New best DETR model saved (AP={ap:.4f})")
    else:
        print(
            f"[HF-DETR] Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"no detections on val set"
        )
        detr_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    torch.save(
        model.state_dict(),
        MODELS_DIR / f"detr_hf_epoch{epoch + 1}.pth",
    )

@torch.no_grad()
def measure_fps_detr_hf(model, processor, loader, device, warmup=20, max_images=500):
    """
    Measure DETR inference FPS (including HF processor preprocessing).

    - model: DetrForObjectDetection
    - processor: DetrImageProcessor
    - loader: DataLoader yielding (images, targets)
    - device: torch.device
    """
    model.eval()

    # ---------- WARMUP (not timed) ----------
    it = iter(loader)
    seen = 0
    for images, _targets in it:
        # images is a tuple/list of tensors in [0,1]
        encoding = processor(
            images=list(images),
            return_tensors="pt",
            do_rescale=False,  # VERY important: same as in training!
        )
        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        _ = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        seen += len(images)
        if seen >= warmup:
            break

    # ---------- TIMED RUN ----------
    num_images = 0
    it = iter(loader)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for images, _targets in it:
        encoding = processor(
            images=list(images),
            return_tensors="pt",
            do_rescale=False,
        )
        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        _ = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        num_images += len(images)
        if num_images >= max_images:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    fps = num_images / elapsed if elapsed > 0 else 0.0
    return fps

fps_detr = [measure_fps_detr_hf(model, processor, val_loader, device)]
print("DETR FPS (val subset):", fps_detr)
with open(RESULTS_DIR / "detr_fps.json", "w") as f:
    json.dump(fps_detr, f, indent=2)


# Save DETR history for plotting
with open(RESULTS_DIR / "detr_hf_history.json", "w") as f:
    json.dump(detr_history, f, indent=2)
print("Saved HF-DETR history to", RESULTS_DIR / "detr_hf_history.json")
