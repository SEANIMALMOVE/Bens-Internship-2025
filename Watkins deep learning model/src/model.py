import torch
import torch.nn as nn
from torchvision import models # type: ignore


class EfficientNetSpectrogram(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # Optionally freeze feature extractor
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x shape: [B, 1, H, W]  (mel spectrogram)
        """
        # Convert 1-channel spectrogram to 3 channels
        x = x.repeat(1, 3, 1, 1)

        # Forward through EfficientNet
        return self.backbone(x)
