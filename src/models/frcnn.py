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
