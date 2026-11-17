"""EfficientNet-B0 for 1-channel spectrograms (same as 2025 / REPSOL)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models


class EfficientNetSpectrogram(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        # ImageNet-pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for p in self.backbone.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        """x: [B, 1, H, W] spectrogram (repeated to 3 channels) or [B, 3, H, W]."""
        if x.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def get_model(model_name: str, num_classes: int, freeze_backbone: bool = False):
    if model_name.lower() in ("efficientnet", "efficientnet_b0", "b0"):
        return EfficientNetSpectrogram(num_classes=num_classes, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model_name: {model_name}")


def get_weighted_criterion(labels, num_classes: int, device=None, label_smoothing: float = 0.0):
    """CrossEntropyLoss with 'balanced' class weights from the training labels."""
    w = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=np.asarray(labels))
    w = torch.tensor(w, dtype=torch.float)
    return nn.CrossEntropyLoss(weight=w.to(device) if device is not None else w,
                               label_smoothing=label_smoothing)
