# Replicating Modern Object Detection Baselines on COCO

This repository contains the code for our deep learning final project **“Replikasi dan Analisis Baseline Modern untuk Deteksi Objek pada COCO Dataset”**.  
We replicate and compare three modern object detection architectures on the COCO 2017 dataset:

- **Faster R-CNN** (two-stage detector, torchvision)
- **YOLOv8s** (one-stage detector, Ultralytics)
- **HF-DETR** (transformer-based detector, Hugging Face)

On a subset of COCO 2017 (5k train / 1k val images), we study convergence behavior, accuracy vs. speed trade-offs, and performance across object scales.

---

## Repository Structure

```text
.
├── detr.py                  # Training + evaluation script for HF-DETR
├── frcnn.py                 # Training + evaluation script for Faster R-CNN
├── yolo.py                  # Training + evaluation script for YOLOv8s
├── convert_dataset.ipynb    # COCO → YOLO-format subset converter
├── plot.ipynb               # Notebook for reproducing the figures/plots
├── get_data.sh              # Helper script to download COCO 2017
├── train.sh                 # Example script to run all three models
├── my_utils/
│   ├── coco_utils.py        # COCO dataloaders, transforms, and helpers
│   └── seed_utils.py        # Reproducible seeding for Python/NumPy/PyTorch
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT license
└── README.md
```

During training/evaluation the following folders are created automatically:

- `data/` – COCO dataset and YOLO subset (after running `get_data.sh` / `convert_dataset.ipynb`)
- `trained_models/` – saved checkpoints for each model
- `results/` – JSON/CSV logs and FPS measurements used for plotting

---

## Setup

### 1. Environment

Recommended (what we used in the project):

- Python ≥ 3.10 (report uses Python 3.12)
- PyTorch with CUDA support
- Torchvision
- Ultralytics (YOLOv8)
- Transformers (Hugging Face)

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> If you prefer, you can install a matching PyTorch build manually from the official website and then `pip install -r requirements.txt` for the rest.

---

## Dataset: COCO 2017

1. Download COCO 2017 (train/val/test images + annotations) using the helper script:

   ```bash
   bash get_data.sh
   ```

   This will create:

   ```text
   data/coco/
     ├── train2017/
     ├── val2017/
     ├── test2017/
     ├── annotations/
     └── annotations/...
   ```

2. **Adjust dataset paths (important).**

   In the scripts we hard-coded our local COCO path (`/mnt/ssd2/santana-coco/data/coco`).  
   If your dataset is under `data/coco` (the default from `get_data.sh`), edit the following:

   - `frcnn.py` and `detr.py`  
     ```python
     train_loader, val_loader = make_coco_loaders(
         root="data/coco",
         ...
     )
     ```

   - `yolo.py`  
     ```python
     COCO_VAL_ANN = Path("data/coco/annotations/instances_val2017.json")
     ```

   - `convert_dataset.ipynb`  
     ```python
     COCO_ROOT = Path("data/coco")
     OUT_ROOT = Path("data/coco_subset")   # or any folder you like
     ```

   Feel free to point these paths anywhere as long as they match your local COCO installation.

---

## YOLO Subset Conversion

For YOLOv8 we train on a **YOLO-format subset** of COCO.

1. Open **`convert_dataset.ipynb`**.
2. Set the config near the top:

   ```python
   COCO_ROOT = Path("data/coco")          # where COCO lives
   OUT_ROOT  = Path("data/coco_subset")   # output folder

   TRAIN_COUNT = 5000     # number of training images
   VAL_COUNT   = 1000     # number of validation images
   SEED        = 0
   COPY_FILES  = False    # True = copy, False = symlink
   ```

3. Run all cells. This will:

   - Create `images/train`, `images/val`, `labels/train`, `labels/val` under `OUT_ROOT`.
   - Write a YOLO data config: `OUT_ROOT/coco_subset.yaml`.

4. Either:

   - Copy/symlink `coco_subset.yaml` into the repo as `data/coco_subset.yaml`, **or**
   - Edit `yolo.py` to point directly to the YAML generated under `OUT_ROOT`.

---

## Running the Experiments

All scripts assume you are in the repo root and your environment is activated.

### 1. Faster R-CNN (torchvision)

```bash
python frcnn.py
```

This script:

- Builds COCO dataloaders (with optional `train_limit=5000`, `val_limit=1000` inside the script).
- Fine-tunes `fasterrcnn_resnet50_fpn` for 10 epochs.
- Computes full COCO metrics every epoch (AP, AP50, AP75, APs, APm, APl).
- Saves:
  - `trained_models/frcnn_epoch{N}.pth`
  - `results/frcnn_history.json` – per-epoch losses and COCO metrics
  - `results/frcnn_fps.json` – inference FPS on the validation subset

### 2. YOLOv8s (Ultralytics)

```bash
python yolo.py
```

This script:

- Loads `yolov8s.pt` pretrained on COCO.
- Evaluates the pretrained model on the subset validation set (epoch 0).
- Trains for 10 epochs on the YOLO-format subset defined in `data/coco_subset.yaml`.
- After each epoch it:
  - Runs `model.val(...)` to obtain `predictions.json`.
  - Uses `pycocotools` to compute COCO AP (including APs/APm/APl) on the same image subset.
- Saves:
  - YOLO training artifacts under `results/yolo_epoch*/`
  - `results/yolo_history.json` – consolidated per-epoch metrics
  - `results/yolo_fps.json` – FPS on the validation subset

### 3. HF-DETR (Hugging Face Transformers)

```bash
python detr.py
```

This script:

- Uses `DetrForObjectDetection` + `DetrImageProcessor` from `transformers`.
- Builds COCO dataloaders (same subset as Faster R-CNN).
- Fine-tunes DETR for 10 epochs.
- Evaluates COCO AP each epoch via `pycocotools`.
- Saves:
  - `trained_models/detr_epoch{N}.pth`
  - `results/detr_hf_history.json` – per-epoch losses and metrics
  - `results/detr_fps.json` – FPS on the validation subset

### 4. Run Everything at Once

If you already have a virtual environment at `./venv`, you can use:

```bash
bash train.sh
```

This activates the venv, runs `frcnn.py`, `yolo.py`, and `detr.py` sequentially, then deactivates the venv.

---

## Reproducing Figures

The **`plot.ipynb`** notebook loads the JSON logs from `results/` and reproduces the main figures:

- Best vs. final epoch AP and validation loss across models
- Training/validation loss curves
- AP curves per epoch (overall and per object size)
- Additional plots used for the final report

Simply open the notebook and run all cells after your experiments finish.

---

## Summary of Results (COCO Subset)

On a subset of COCO 2017 (5k train / 1k val images, 10 training epochs):

- **YOLOv8s**
  - mAP@[0.50:0.95] ≈ **33%**
  - Inference speed ≈ **246 FPS**
- **Faster R-CNN**
  - mAP@[0.50:0.95] ≈ **30%**
  - Inference speed ≈ **17 FPS**
- **HF-DETR**
  - mAP@[0.50:0.95] ≈ **6–7%** (failed to converge with our limited compute)
  - Inference speed ≈ **23 FPS**

These numbers highlight a clear accuracy–speed trade-off and show that DETR is much more sensitive to hyperparameters and dataset size on this small subset.

---

## Citation

If you find this code useful in your work, please cite our project report:

```text
Pradata, S. Y., Kurniawan, Y. R., Mukhaer, A. A., Adnan, K. A., & Widja, A. R.
"Replikasi dan Analisis Baseline Modern untuk Deteksi Objek pada COCO Dataset", 2024.
Departemen Ilmu Komputer dan Elektronika, Universitas Gadjah Mada.
```

(You can also link directly to this repository in your bibliography as
“Implementation code for object detection baseline comparison”.)
