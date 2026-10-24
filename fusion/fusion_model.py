from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import resnet50


def load_branch_a_backbone(
    checkpoint_path: Union[str, Path] = "checkpoints/imagenet_ai_0419_biggan/best_model.pt",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, nn.Module]:
    """Load Branch A ResNet50 spatial detector.

    Returns:
        (backbone, head) where backbone outputs 2048-dim features and head outputs single logit.
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Branch A checkpoint not found: {path}")

    model = resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, 1))

    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if device:
        model = model.to(device)

    # Separate feature extractor (everything up to avgpool) from classification head
    head = model.fc
    model.fc = nn.Identity()
    return model, head


def load_branch_b_backbone(
    checkpoint_path: Union[str, Path] = "branch_b/models/frequency/best_model.pt",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, nn.Module]:
    """Load Branch B ResNet50 frequency detector.

    Returns:
        (backbone, head) where backbone outputs 2048-dim features and head outputs 2 logits.
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Branch B checkpoint not found: {path}")

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if device:
        model = model.to(device)

    head = model.fc
    model.fc = nn.Identity()
    return model, head


class FeatureFusionModel(nn.Module):
    """Dual-branch feature concatenation fusion model (Spatial + Frequency)."""

    def __init__(
        self,
        branch_a_checkpoint: str = "checkpoints/imagenet_ai_0419_biggan/best_model.pt",
        branch_b_checkpoint: str = "branch_b/models/frequency/best_model.pt",
        freeze_backbones: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.freeze_backbones = freeze_backbones

        # Backbones
        self.spatial_backbone, self.spatial_head = load_branch_a_backbone(branch_a_checkpoint)
        self.frequency_backbone, self.frequency_head = load_branch_b_backbone(branch_b_checkpoint)

        if freeze_backbones:
            for param in self.spatial_backbone.parameters():
                param.requires_grad = False
            for param in self.frequency_backbone.parameters():
                param.requires_grad = False
            self.spatial_backbone.eval()
            self.frequency_backbone.eval()

        # Fusion classification head: maps 4096 (2048 + 2048) -> 2 logits
        self.fusion_head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        spatial_images: torch.Tensor,
        freq_images: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            spatial_images: Tensor of shape (B, 3, 224, 224)
            freq_images: Tensor of shape (B, 3, 224, 224)

        Returns:
            Logits of shape (B, 2) where 0=real, 1=ai
        """
        if self.freeze_backbones:
            with torch.no_grad():
                feat_spatial = self.spatial_backbone(spatial_images)
                feat_freq = self.frequency_backbone(freq_images)
        else:
            feat_spatial = self.spatial_backbone(spatial_images)
            feat_freq = self.frequency_backbone(freq_images)

        # Concatenate features along feature dimension
        combined_features = torch.cat([feat_spatial, feat_freq], dim=1)  # (B, 4096)
        logits = self.fusion_head(combined_features)
        return logits


class LateFusionModel:
    """Late-fusion decision ensemble combining Branch A and Branch B probabilities."""

    def __init__(
        self,
        branch_a_checkpoint: str = "checkpoints/imagenet_ai_0419_biggan/best_model.pt",
        branch_b_checkpoint: str = "branch_b/models/frequency/best_model.pt",
        alpha: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.alpha = float(alpha)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Branch A
        self.model_a = resnet50(weights=None)
        self.model_a.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(self.model_a.fc.in_features, 1))
        ckpt_a = torch.load(branch_a_checkpoint, map_location=self.device, weights_only=False)
        state_a = ckpt_a["model_state_dict"] if "model_state_dict" in ckpt_a else ckpt_a
        self.model_a.load_state_dict(state_a)
        self.model_a.to(self.device).eval()

        # Load Branch B
        self.model_b = resnet50(weights=None)
        self.model_b.fc = nn.Linear(self.model_b.fc.in_features, 2)
        ckpt_b = torch.load(branch_b_checkpoint, map_location=self.device, weights_only=False)
        state_b = ckpt_b["model_state_dict"] if "model_state_dict" in ckpt_b else ckpt_b
        self.model_b.load_state_dict(state_b)
        self.model_b.to(self.device).eval()

    @torch.no_grad()
    def predict_probs(
        self,
        spatial_images: torch.Tensor,
        freq_images: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute individual and fused probabilities for class 1 (AI).

        Returns:
            (prob_fused, prob_a, prob_b)
        """
        w = self.alpha if alpha is None else float(alpha)

        spatial_images = spatial_images.to(self.device)
        freq_images = freq_images.to(self.device)

        # Branch A logit -> sigmoid -> prob_ai
        logit_a = self.model_a(spatial_images).squeeze(-1)
        prob_a = torch.sigmoid(logit_a)

        # Branch B logits -> softmax -> prob_ai (class index 1)
        logits_b = self.model_b(freq_images)
        prob_b = torch.softmax(logits_b, dim=1)[:, 1]

        # Fused probability: convex combination
        prob_fused = w * prob_a + (1.0 - w) * prob_b
        return prob_fused, prob_a, prob_b
