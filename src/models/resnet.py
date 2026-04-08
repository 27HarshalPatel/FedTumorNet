"""CNN backbone architectures for Brain Tumor Classification."""
import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes=4, pretrained=True):
    """ResNet-50 with ImageNet weights; replaces the final FC layer."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(2048, num_classes)
    return model

def get_efficientnet_b0(num_classes=4, pretrained=True):
    """EfficientNet-B0 for ablation experiments."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_vit_small(num_classes=4, pretrained=True):
    """ViT-B/16 as ViT-Small proxy (torchvision)."""
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

def get_model(model_name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Factory for backbone models."""
    registry = {
        "resnet50":       get_resnet50,
        "efficientnet_b0": get_efficientnet_b0,
        "vit_small":      get_vit_small,
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(registry)}")
    return registry[model_name](num_classes=num_classes, pretrained=pretrained)
