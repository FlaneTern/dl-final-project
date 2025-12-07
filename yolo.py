from my_utils.seed_utils import set_seed
set_seed(42)   # call once at the very top, before dataloaders/models

from pathlib import Path
import json
import csv


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.data.converter import coco80_to_coco91_class


# ============================================================
# Common setup: paths for models and results
# ============================================================
MODELS_DIR = Path("trained_models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# COCO-style annotation JSON for val subset
COCO_VAL_ANN = Path("/mnt/ssd2/santana-coco/data/coco/annotations/instances_val2017.json")

# Number of training epochs
EPOCHS = 10


# ============================================================
# Helper: COCO metrics including APs / APm / APl
# ============================================================
def coco_area_metrics(ann_json: Path, pred_json: Path):
    """
    Run COCOeval on YOLO predictions.json and return:
      AP (0.5:0.95), AP50, AP75, APs, APm, APl

    Handles two important details:

      1) YOLOv8 (on non-COCO datasets) outputs category_id in 1..nc, while
         official COCO annotations use a sparse 91-ID scheme. We map 1..80
         -> COCO 91 IDs using coco80_to_coco91_class().

      2) If you're evaluating a *subset* of COCO (e.g. coco_subset.yaml),
         predictions.json only contains some image_ids. We restrict
         COCOeval to those imgIds so AP is computed on the same subset,
         not diluted over all val2017 images.
    """
    coco_gt = COCO(str(ann_json))

    # ---- Load predictions ----
    with pred_json.open("r") as f:
        preds = json.load(f)

    if not preds:
        raise ValueError(f"No detections found in {pred_json}")

    # Collect image_ids present in the predictions (for subset eval)
    img_ids = sorted({int(det["image_id"]) for det in preds})

    # Detect category_id range
    cat_ids = {int(det["category_id"]) for det in preds}
    max_cat = max(cat_ids)

    # Default: evaluate on the given predictions file
    pred_for_eval = pred_json

    # If category_id range is <= 80, assume YOLO's 1..80 space and map to COCO 91 IDs
    if max_cat <= 80:
        coco_map = coco80_to_coco91_class()  # list of length 80

        for det in preds:
            c = int(det["category_id"])
            if 1 <= c <= 80:
                det["category_id"] = coco_map[c - 1]

        # Write a remapped JSON file next to the original
        pred_for_eval = pred_json.with_name(pred_json.stem + "_coco91.json")
        with pred_for_eval.open("w") as f:
            json.dump(preds, f)

    # ---- Standard COCOeval, but restricted to our subset image_ids ----
    coco_dt = coco_gt.loadRes(str(pred_for_eval))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # Restrict evaluation to the image IDs actually present in predictions.json
    coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    s = coco_eval.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    return {
        "AP":   float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "APs":  float(s[3]),
        "APm":  float(s[4]),
        "APl":  float(s[5]),
    }
# ============================================================
# YOLOv8: train + per-epoch eval
# ============================================================
yolo_model = YOLO("yolov8s.pt")  # pretrained on full COCO

history = []  # will become yolo_history.json


# ------------------------------------------------------------
# (A) Baseline: pretrained YOLO on val subset (epoch = 0)
# ------------------------------------------------------------
print("Evaluating YOLOv8s (pretrained) on coco_subset val...")

baseline_metrics = yolo_model.val(
    data="data/coco_subset.yaml",
    split="val",
    save_json=True,                           # needed for predictions.json
    project=str(RESULTS_DIR / "yolo_epoch"),  # all epoch evals go here
    name="epoch0_pretrained",
)

pred_json0 = Path(baseline_metrics.save_dir) / "predictions.json"
if COCO_VAL_ANN.exists() and pred_json0.exists():
    base_coco = coco_area_metrics(COCO_VAL_ANN, pred_json0)
    history.append({
        "epoch": 0,
        "train_loss": None,
        "val_loss": None,
        "AP":  base_coco["AP"],
        "AP50": base_coco["AP50"],
        "AP75": base_coco["AP75"],
        "APs":  base_coco["APs"],
        "APm":  base_coco["APm"],
        "APl":  base_coco["APl"],
    })
else:
    raise FileNotFoundError(
        "Baseline predictions.json or COCO_VAL_ANN not found; "
        "cannot compute epoch 0 COCO metrics."
    )


# ------------------------------------------------------------
# (B) Train on subset COCO
#      - save_period=1 so we have epoch0.pt, epoch1.pt, ...
# ------------------------------------------------------------
print("Training YOLOv8s on coco_subset...")

train_results = yolo_model.train(
    data="data/coco_subset.yaml",
    epochs=EPOCHS,
    imgsz=640,
    batch=16,          # shrink if OOM
    device=0,          # or "cuda:0"
    workers=4,
    project=str(MODELS_DIR / "yolo"),
    name="yolov8s_640_subset",
    save_period=1,     # <<< save weights every epoch: epoch0.pt, epoch1.pt, ...
)

train_run_dir = Path(train_results.save_dir)    # e.g. trained_models/yolo/yolov8s_640_subset
results_csv = train_run_dir / "results.csv"


# ------------------------------------------------------------
# (C) Parse results.csv to get per-epoch train_loss / val_loss
# ------------------------------------------------------------
if not results_csv.exists():
    raise FileNotFoundError(f"results.csv not found at {results_csv}")

epoch_losses = {}  # epoch_idx (0-based) -> (train_loss, val_loss)

with results_csv.open("r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        epoch_idx = int(row["epoch"])

        def fget(key):
            v = row.get(key, "")
            if v in ("", None):
                return 0.0
            try:
                return float(v)
            except ValueError:
                return 0.0

        # Sum components into a single total train and val loss
        train_loss = (
            fget("train/box_loss") +
            fget("train/cls_loss") +
            fget("train/dfl_loss")
        )
        val_loss = (
            fget("val/box_loss") +
            fget("val/cls_loss") +
            fget("val/dfl_loss")
        )

        epoch_losses[epoch_idx] = (train_loss, val_loss)

# ------------------------------------------------------------
# (D) For each training epoch: run COCOeval and add to history
#     - history epoch 0 = pretrained baseline (already added)
#     - history epoch k  = the checkpoint for epoch k
# ------------------------------------------------------------
metrics_last = None
per_epoch_eval_project = RESULTS_DIR / "yolo_epoch"

if not epoch_losses:
    raise RuntimeError("No epoch losses parsed from results.csv; check CSV content.")

min_epoch_idx = min(epoch_losses.keys())
max_epoch_idx = max(epoch_losses.keys())
print(f"[INFO] epoch indices from CSV: {sorted(epoch_losses.keys())}")
print(f"[INFO] min_epoch_idx={min_epoch_idx}, max_epoch_idx={max_epoch_idx}")

# Map CSV epoch indices to checkpoint “epochX.pt” numbers.
# If CSV starts at 0, checkpoints are epoch1.pt, epoch2.pt, ...
# If CSV starts at 1, checkpoints are epoch1.pt, epoch2.pt, ...
if min_epoch_idx == 0:
    def csv_to_ckpt_epoch(epoch_idx: int) -> int:
        return epoch_idx + 1
else:
    def csv_to_ckpt_epoch(epoch_idx: int) -> int:
        return epoch_idx

weights_dir = train_run_dir / "weights"
last_ckpt_path = weights_dir / "last.pt"

for epoch_idx in sorted(epoch_losses.keys()):
    ckpt_epoch = csv_to_ckpt_epoch(epoch_idx)
    ckpt = weights_dir / f"epoch{ckpt_epoch}.pt"

    # If this is the final epoch and epoch{ckpt_epoch}.pt doesn't exist,
    # fall back to last.pt (Ultralytics saves it by default).
    if not ckpt.exists():
        if epoch_idx == max_epoch_idx and last_ckpt_path.exists():
            print(f"[INFO] epoch{ckpt_epoch}.pt not found for final epoch; "
                  f"using last.pt as epoch {ckpt_epoch}.")
            ckpt = last_ckpt_path
        else:
            print(f"[WARN] Skipping CSV epoch_idx={epoch_idx} "
                  f"(ckpt_epoch={ckpt_epoch}): checkpoint {ckpt} not found.")
            continue

    print(f"Evaluating checkpoint for epoch {ckpt_epoch} on val (COCO metrics)...")
    model_e = YOLO(str(ckpt))
    metrics_e = model_e.val(
        data="data/coco_subset.yaml",
        split="val",
        save_json=True,
        project=str(per_epoch_eval_project),
        name=f"epoch{ckpt_epoch}",
        verbose=False,
    )
    metrics_last = metrics_e  # for FPS later

    pred_json_e = Path(metrics_e.save_dir) / "predictions.json"
    if not (COCO_VAL_ANN.exists() and pred_json_e.exists()):
        print(f"[WARN] predictions.json or COCO_VAL_ANN missing "
              f"for ckpt_epoch={ckpt_epoch}; skipping.")
        continue

    area = coco_area_metrics(COCO_VAL_ANN, pred_json_e)
    train_loss, val_loss = epoch_losses[epoch_idx]

    history.append({
        "epoch": ckpt_epoch,  # 1..EPOCHS (0 is baseline)
        "train_loss": train_loss,
        "val_loss": val_loss,
        "AP":   area["AP"],
        "AP50": area["AP50"],
        "AP75": area["AP75"],
        "APs":  area["APs"],
        "APm":  area["APm"],
        "APl":  area["APl"],
    })





# ------------------------------------------------------------
# (E) Save yolo_history.json (only this + FPS as requested)
# ------------------------------------------------------------
yolo_history_path = RESULTS_DIR / "yolo_history.json"
with yolo_history_path.open("w") as f:
    json.dump(history, f, indent=2)
print("Saved YOLO history to", yolo_history_path)


# ------------------------------------------------------------
# (F) FPS from the last per-epoch val (single number JSON)
# ------------------------------------------------------------
if metrics_last is None:
    # Fallback: use baseline metrics if something went wrong
    metrics_last = baseline_metrics

infer_ms = float(metrics_last.speed["inference"])   # ms per image on val
fps_yolo = 1000.0 / infer_ms

fps_path = RESULTS_DIR / "yolo_fps.json"
with fps_path.open("w") as f:
    # Just a single number (float) in the JSON file
    json.dump(fps_yolo, f, indent=2)

print("YOLOv8 FPS (val subset):", fps_yolo)
print("Saved FPS to", fps_path)
