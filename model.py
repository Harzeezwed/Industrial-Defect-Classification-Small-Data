"""
Model definition: an ImageNet-pretrained EfficientNet-B0 backbone with a
small custom classification head, fine-tuned for defect classification
from very few labeled examples per class.

EfficientNet-B0 was chosen because its ImageNet features transfer well
with limited target-domain data, and its parameter count (~5M) keeps
fine-tuning fast enough to iterate quickly in a small-sample setting.
"""
import torch.nn as nn
from torchvision import models


def get_model(num_classes: int, dropout: float = 0.3, hidden_dim: int = 512) -> nn.Module:
    """Build an EfficientNet-B0 classifier for `num_classes` defect types."""
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Linear(hidden_dim, num_classes),
    )
    return model
