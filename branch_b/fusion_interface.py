"""Branch B Frequency Detector Inference & Fusion Interface.

Provides a clean, standalone inference interface for the Branch B FFT detector
to be loaded alongside Branch A without any cross-branch dependencies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from branch_b.fft_utils import image_to_fft
from branch_b.frequency_model import build_frequency_model


class BranchBFrequencyDetector:
    """Wrapper for Branch B (FFT-spectrum ResNet50 classifier)."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "branch_b/models/frequency/best_model.pt",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Build ResNet50 architecture
        self.model = build_frequency_model(pretrained=False).to(self.device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.checkpoint_epoch = checkpoint.get("epoch")
            self.checkpoint_accuracy = checkpoint.get("val_accuracy")
        else:
            state_dict = checkpoint
            self.checkpoint_epoch = None
            self.checkpoint_accuracy = None

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def preprocess(self, image: Image.Image, size: int = 224) -> torch.Tensor:
        """Apply FFT transformation and standard ImageNet normalization."""
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        fft_pil = image_to_fft(image, size=size)
        tensor = self.transform(fft_pil)
        return tensor

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> Dict[str, Union[float, int, List[float]]]:
        """Predict whether a PIL image is real (0) or AI-generated (1).

        Returns:
            Dict with keys:
                - prob_ai: probability of AI/fake (class 1)
                - prob_real: probability of nature/real (class 0)
                - pred_label: 0 (real) or 1 (AI)
                - logits: raw logits [logit_real, logit_ai]
        """
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_label = int(logits.argmax(dim=1).item())

        return {
            "prob_ai": float(probs[1]),
            "prob_real": float(probs[0]),
            "pred_label": pred_label,
            "logits": logits.squeeze(0).cpu().numpy().tolist(),
        }

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract pre-classifier 2048-dim feature vector for fusion.

        Extracts activations from model backbone after global average pooling.
        """
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        x = self.model.conv1(tensor)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1).squeeze(0)  # Shape: (2048,)
        return features
