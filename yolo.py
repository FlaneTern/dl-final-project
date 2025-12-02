# %% [markdown]
# Common setup: paths for models and results

from pathlib import Path
import json
import shutil

MODELS_DIR = Path("trained_models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# %% 
# ======================================
# 2. YOLOv8: train + val (Ultralytics)
# ======================================
from ultralytics import YOLO

# Use the same subset COCO yaml you built earlier
# (assumed to point to subset train/val COCO folders)
yolo_model = YOLO("yolov8s.pt")  # pretrained on full COCO

yolo_history = []

# Optional: evaluate pretrained YOLO before fine-tuning on your subset
print("Evaluating YOLOv8s (pretrained) on coco_subset val...")
metrics0 = yolo_model.val(data="data/coco_subset.yaml", split="val")
pre_entry = {
    "stage": "pretrained",
    "map":   float(metrics0.box.map),
    "map50": float(metrics0.box.map50),
    "map75": float(metrics0.box.map75),
}
yolo_history.append(pre_entry)
print("YOLOv8s initial:", pre_entry)

# Train on subset COCO
yolo_results = yolo_model.train(
    data="data/coco_subset.yaml",
    epochs=10,
    imgsz=640,
    batch=16,          # shrink if OOM
    device=0,          # or "cuda:0"
    workers=4,
    project=str(MODELS_DIR / "yolo"),
    name="yolov8s_640_subset",
)

# Validation on subset val (consistent with proposal setup)
metrics = yolo_model.val(data="data/coco_subset.yaml", split="val")
post_entry = {
    "stage": "finetuned",
    "map":   float(metrics.box.map),
    "map50": float(metrics.box.map50),
    "map75": float(metrics.box.map75),
}

# per-class mAP50-95 (for extra analysis if needed)
try:
    post_entry["per_class_map"] = [float(x) for x in metrics.box.maps]
except Exception:
    post_entry["per_class_map"] = None

yolo_history.append(post_entry)
print("YOLOv8s fine-tuned:", post_entry)

# Save YOLO history JSON for plotting
with open(RESULTS_DIR / "yolo_history.json", "w") as f:
    json.dump(yolo_history, f, indent=2)
print("Saved YOLO history to", RESULTS_DIR / "yolo_history.json")

# Copy Ultralytics training results.csv into results/ for easy loading
yolo_run_dir = MODELS_DIR / "yolo" / "yolov8s_640_subset"
yolo_results_csv = yolo_run_dir / "results.csv"
if yolo_results_csv.exists():
    shutil.copy2(yolo_results_csv, RESULTS_DIR / "yolov8_results.csv")
    print("Copied YOLO results.csv to", RESULTS_DIR / "yolov8_results.csv")
else:
    print("WARNING: YOLO results.csv not found at", yolo_results_csv)

# Best YOLO weights are already in:
#   trained_models/yolo/yolov8s_640_subset/weights/best.pt
# which matches your 'trained_models/' requirement.
