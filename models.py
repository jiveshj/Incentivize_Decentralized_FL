"""
Model architectures as specified in the paper.

- FashionMNIST: Deep MLP with two hidden layers (64, 30) + dropout
- EMNIST: LeNet-5
- CIFAR-10: ResNet-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CelebaSmileCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 64x64 -> 32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # 8x8 -> 4x4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2)

        # global average pooling + binary head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 2)  # smiling / not smiling

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)                  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)        # [B, 128]
        x = self.dropout(x)
        x = self.fc(x)
        return x
        
class CifarCNNSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.5)
        # Directly to logits: 32 * 16 * 16 -> num_classes
        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
        
class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # 16x16 -> 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # 8x8 -> 4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DeepMLP(nn.Module):
    """
    Deep MLP for FashionMNIST as described in the paper:
    - Two hidden layers with 64 and 30 units
    - Dropout after first hidden layer
    - Input: flattened normalized 28x28 image
    - Output: 10 classes
    """
    def __init__(self, input_dim: int = 784, num_classes: int = 10,
                 hidden1: int = 64, hidden2: int = 30, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    """
    LeNet-5 for EMNIST.
    Adapted for 28x28 grayscale input, 47 output classes.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 47):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet18Cifar(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10.
    - Replaces first 7x7 conv with 3x3
    - Removes initial maxpool
    - Adjusts final FC for 10 classes
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = resnet18(weights=None, num_classes=num_classes)
        # Replace first conv: 7x7 stride 2 -> 3x3 stride 1 for 32x32 inputs
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        # Remove maxpool
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_model_fn(dataset_name: str, num_classes: int = None):
    """
    Return a model factory function for the given dataset.
    Matches the paper's model choices.

    Returns:
        callable that returns a fresh nn.Module
    """
    if dataset_name == "fashionmnist":
        nc = num_classes or 10
        return lambda: DeepMLP(input_dim=784, num_classes=nc)

    elif dataset_name == "emnist":
        nc = num_classes or 47
        return lambda: LeNet5(in_channels=1, num_classes=nc)

    elif dataset_name == "cifar10":
        nc = num_classes or 10
        return lambda: CifarCNN(num_classes=nc)

    elif dataset_name == "cifar100":
        nc = num_classes or 100
        return lambda: CifarCNN(num_classes=nc)

    elif dataset_name == "mnist":
        nc = num_classes or 10
        return lambda: DeepMLP(input_dim=784, num_classes=nc)

    elif dataset_name == "celeba":
        return lambda: CelebaSmileCNN()

    else:
        raise ValueError(f"No default model for dataset '{dataset_name}'")
