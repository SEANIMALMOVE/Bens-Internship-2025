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



# Define a simple CNN model for spectrogram classification
class BaselineCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()

        # extracts features from spectrogram
        self.features = nn.Sequential(
            # low level patterns (edges, textures)
            nn.Conv2d(input_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), 
            # more complex patterns (harmonics, frequency bands)
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # high level structures (shapes of vocyyalizations)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), 
        )   
            ### -> 1-16-32-64 the model learns more complex feature maps
            # MaxPool2d = width decreases by factor of 2 each time
            # height also gets reduced 
            ### -> [batch_size, 64, 16, reduced_width]


        # converts extracted features to class decisions   
        self.classifier = nn.Sequential(
            nn.Flatten(), # -> turns feature map into vector

            # nn.Linear(64 *  (64) * (64), 128),  # size of input spectrogram 
            nn.LazyLinear(128), # automatically

            nn.ReLU(), # non-linear activation
            nn.Dropout(0.4), # randomly turn off 40% of neurons during training to prevent overfitting
            nn.Linear(128, num_classes) # output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x