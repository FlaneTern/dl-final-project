import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import random

class CocoDetectionWrapper(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # img: PIL Image, target: list[dict]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in target:
            xmin, ymin, w, h = obj["bbox"]

            # ---- filter out degenerate boxes directly from COCO ----
            if w <= 0 or h <= 0:
                continue

            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj["category_id"])
            areas.append(obj["area"])
            iscrowd.append(obj.get("iscrowd", 0))

        if len(boxes) == 0:
            # no valid boxes left → empty targets
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

            # extra safety: in case of weird float rounding from COCO
            wh = boxes[:, 2:] - boxes[:, :2]  # (N, 2) = [w, h]
            keep = (wh[:, 0] > 0) & (wh[:, 1] > 0)

            boxes = boxes[keep]
            labels = labels[keep]
            areas = areas[keep]
            iscrowd = iscrowd[keep]

            if boxes.numel() == 0:
                # everything got filtered out → treat as no-annotation image
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                areas = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)

        coco_img_id = self.ids[idx]
        
        
        target_out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(coco_img_id, dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        # apply your transforms (can be augmentations)
        if self._transforms is not None:
            img = self._transforms(img)

        # ensure tensor image
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)

        return img, target_out


def collate_fn(batch):
    return tuple(zip(*batch))


def make_coco_loaders(
    root="data/coco",
    batch_size=2,
    num_workers=4,
    train_limit=None,   # e.g. 5000
    val_limit=None,     # e.g. 1000
    subset_random=True, # random subset instead of first N
):
    train_img = f"{root}/train2017"
    train_ann = f"{root}/annotations/instances_train2017.json"
    val_img = f"{root}/val2017"
    val_ann = f"{root}/annotations/instances_val2017.json"

    def base_transform(img):
        return img  # or add augmentations here

    train_full = CocoDetectionWrapper(train_img, train_ann, transforms=base_transform)
    val_full = CocoDetectionWrapper(val_img, val_ann, transforms=base_transform)

    # ----- build subsets -----
    def make_subset(ds_full, limit):
        if limit is None or limit >= len(ds_full):
            return ds_full
        indices = list(range(len(ds_full)))
        if subset_random:
            random.seed(0)  # for reproducibility
            random.shuffle(indices)
        indices = indices[:limit]
        return Subset(ds_full, indices)

    train_ds = make_subset(train_full, train_limit)
    val_ds = make_subset(val_full, val_limit)
    # -------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
