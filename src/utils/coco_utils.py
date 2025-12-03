# src/utils/coco_utils.py

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import torchvision.transforms as T


# -------------------------
# Detection-aware transforms
# -------------------------

class DetCompose:
    """Compose transforms that operate on (img, target)."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class DetRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target

        # get width
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            w, h = img.size

        img = F.hflip(img)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            # x1, y1, x2, y2 -> flip horizontally
            xmin = boxes[:, 0].clone()
            xmax = boxes[:, 2].clone()
            boxes[:, 0] = w - xmax
            boxes[:, 2] = w - xmin
            target["boxes"] = boxes

        return img, target


class DetRandomCrop:
    """
    Simple random crop that also updates boxes.

    scale: tuple(min_scale, max_scale) relative to original size
    p: probability of applying crop
    """
    def __init__(self, scale=(0.8, 1.0), p=0.5):
        self.scale = scale
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target

        # get original H, W
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            w, h = img.size

        min_s, max_s = self.scale
        scale = random.uniform(min_s, max_s)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # in case rounding gives same size
        new_h = max(1, min(new_h, h))
        new_w = max(1, min(new_w, w))
        if new_h == h and new_w == w:
            return img, target

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        # crop image
        img = F.crop(img, top, left, new_h, new_w)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            # translate boxes
            boxes[:, [0, 2]] = boxes[:, [0, 2]] - left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] - top

            # clamp to [0, new_w/h]
            boxes[:, 0].clamp_(min=0, max=new_w)
            boxes[:, 2].clamp_(min=0, max=new_w)
            boxes[:, 1].clamp_(min=0, max=new_h)
            boxes[:, 3].clamp_(min=0, max=new_h)

            # drop boxes that became too small / invalid
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            keep = (ws > 1) & (hs > 1)

            boxes = boxes[keep]
            target["boxes"] = boxes
            target["labels"] = target["labels"][keep]
            if "iscrowd" in target:
                target["iscrowd"] = target["iscrowd"][keep]
            if "area" in target:
                target["area"] = target["area"][keep]  # will be recomputed anyway

        return img, target


class DetColorJitter:
    """Color jitter on the image only."""
    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.8,
    ):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = self.jitter(img)
        return img, target


class DetToTensor:
    """Convert PIL -> Tensor, keep target as-is."""
    def __call__(self, img, target):
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        return img, target


# -------------------------
# COCO dataset wrapper
# -------------------------

class CocoDetectionWrapper(CocoDetection):
    """
    Wraps CocoDetection to output:
      img: Tensor[C, H, W]
      target: dict with fields:
        - boxes (FloatTensor[N, 4], xyxy)
        - labels (Int64Tensor[N])
        - image_id (Int64Tensor[1])
        - area (FloatTensor[N])
        - iscrowd (Int64Tensor[N])
    """

    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # img: PIL, target: list[dict]

        boxes = []
        labels = []
        iscrowd = []

        for obj in target:
            xmin, ymin, w, h = obj["bbox"]

            # filter degenerate boxes from COCO
            if w <= 0 or h <= 0:
                continue

            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj["category_id"])
            iscrowd.append(obj.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

            # extra safety
            wh = boxes[:, 2:] - boxes[:, :2]
            keep = (wh[:, 0] > 0) & (wh[:, 1] > 0)
            boxes = boxes[keep]
            labels = labels[keep]
            iscrowd = iscrowd[keep]

            if boxes.numel() == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                iscrowd = torch.zeros((0,), dtype=torch.int64)

        coco_img_id = self.ids[idx]

        target_out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(coco_img_id, dtype=torch.int64),
            "iscrowd": iscrowd,
        }

        # apply detection-aware transforms (can include flips, crops, jitter, to_tensor)
        if self._transforms is not None:
            img, target_out = self._transforms(img, target_out)
        else:
            # at least ensure tensor image
            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)

        # ensure final types, and recompute area based on final boxes
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)

        boxes = target_out["boxes"]
        labels = target_out["labels"]
        iscrowd = target_out["iscrowd"]

        boxes = boxes.to(torch.float32)
        labels = labels.to(torch.int64)
        iscrowd = iscrowd.to(torch.int64)

        if boxes.numel() > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            areas = torch.zeros((0,), dtype=torch.float32)

        target_out["boxes"] = boxes
        target_out["labels"] = labels
        target_out["iscrowd"] = iscrowd
        target_out["area"] = areas

        return img, target_out


def collate_fn(batch):
    return tuple(zip(*batch))


# -------------------------
# make_coco_loaders: now with proper aug
# -------------------------

def make_coco_loaders(
    root="data/coco",
    batch_size=2,
    num_workers=4,
    train_limit=None,   # None => full train2017
    val_limit=None,     # None => full val2017
    subset_random=True, # random subset instead of first N
):
    root = Path(root)
    train_img = root / "train2017"
    train_ann = root / "annotations/instances_train2017.json"
    val_img = root / "val2017"
    val_ann = root / "annotations/instances_val2017.json"

    # TRAIN: random flip, random crop, color jitter, then ToTensor
    train_transforms = DetCompose([
        DetRandomHorizontalFlip(p=0.5),
        DetRandomCrop(scale=(0.8, 1.0), p=0.5),
        DetColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.8,
        ),
        DetToTensor(),
    ])

    # VAL: only convert to tensor (no random aug)
    val_transforms = DetCompose([
        DetToTensor(),
    ])

    train_full = CocoDetectionWrapper(str(train_img), str(train_ann),
                                      transforms=train_transforms)
    val_full = CocoDetectionWrapper(str(val_img), str(val_ann),
                                    transforms=val_transforms)

    def make_subset(ds_full, limit):
        if limit is None or limit >= len(ds_full):
            return ds_full
        indices = list(range(len(ds_full)))
        if subset_random:
            random.shuffle(indices)
        indices = indices[:limit]
        return Subset(ds_full, indices)

    train_ds = make_subset(train_full, train_limit)
    val_ds = make_subset(val_full, val_limit)

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
