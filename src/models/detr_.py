import torch
import sys



def get_detr_model(num_classes=91, pretrained=True):
    if pretrained:
        model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
    else:
        model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=False)

    # If you want to adapt class head:
    if num_classes != 91:
        in_features = model.class_embed.in_features
        model.class_embed = torch.nn.Linear(in_features, num_classes)
        
        
    return model

