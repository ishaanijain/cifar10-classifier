import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # default resnet conv1 is 7x7 stride 2 — way too aggressive for 32x32 cifar images
    # swap it for a smaller 3x3 conv that preserves spatial resolution
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # also skip the maxpool, same reason

    # new classification head for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
