import torch.nn as nn
from torchvision import models


class ResNetTransferLearning(nn.Module):
    """ResNet-18 (ImageNet) с замороженным backbone и новой головой на 10 классов."""

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
