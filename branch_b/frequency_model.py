import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_frequency_model(pretrained=True):
    """Build a ResNet50 classifier for FFT-spectrum images."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    # Two classes: real (0) and AI-generated (1)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model