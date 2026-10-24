from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from branch_b.fft_utils import image_to_fft


class DualStreamDataset(Dataset):
    """Dataset providing both spatial (RGB) and frequency (FFT) views of images."""

    def __init__(
        self,
        csv_path: str,
        image_col: str = "path",
        label_col: str = "label",
        image_size: int = 224,
        generator_col: str = "generator",
        exclude_generators: Sequence[str] = (),
        deduplicate_paths: bool = True,
    ):
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.data = pd.read_csv(self.csv_path)
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = image_size
        self.generator_col = generator_col

        missing = {image_col, label_col} - set(self.data.columns)
        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}. Available: {list(self.data.columns)}"
            )

        self.data = self.data.copy()
        self.data["_fusion_label"] = [
            self._map_label(val, idx + 2)
            for idx, val in enumerate(self.data[label_col])
        ]

        # Handle excluded generators
        excluded = {str(name).strip().lower() for name in exclude_generators}
        self.excluded_row_count = 0
        if excluded:
            if generator_col not in self.data.columns:
                raise ValueError(f"Generator column {generator_col!r} not in CSV.")
            gens = self.data[generator_col].astype(str).str.strip().str.lower()
            excluded_mask = gens.isin(excluded)
            self.excluded_row_count = int(excluded_mask.sum())
            self.data = self.data.loc[~excluded_mask].copy()

        # Handle empty paths
        empty_paths = self.data[image_col].isna() | (
            self.data[image_col].astype(str).str.strip() == ""
        )
        if empty_paths.any():
            row_idx = int(empty_paths[empty_paths].index[0]) + 2
            raise ValueError(f"Empty image path in {self.csv_path}, CSV row {row_idx}.")

        self.data["_fusion_path_key"] = self.data[image_col].map(self._path_key)
        self.duplicate_row_count = 0
        if deduplicate_paths:
            deduped = self.data.drop_duplicates(
                subset="_fusion_path_key",
                keep="first",
            )
            self.duplicate_row_count = len(self.data) - len(deduped)
            self.data = deduped.copy()

        # Transforms for Spatial stream
        self.spatial_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Transforms for Frequency stream (FFT magnitude spectrum)
        self.freq_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        image_path = self._resolve_path(row[self.image_col])
        csv_row = int(row.name) + 2

        try:
            with Image.open(image_path) as src:
                rgb_image = src.convert("RGB")
                # Spatial view
                spatial_tensor = self.spatial_transform(rgb_image)
                # Frequency view
                fft_pil = image_to_fft(rgb_image, size=self.image_size)
                freq_tensor = self.freq_transform(fft_pil)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process image {image_path} (row {csv_row}): {exc}"
            ) from exc

        label = int(row["_fusion_label"])
        return spatial_tensor, freq_tensor, label

    def _map_label(self, raw_label, csv_row: int) -> int:
        lbl = str(raw_label).strip().lower()
        if lbl in {"real", "nature", "0", "0.0"}:
            return 0
        if lbl in {"ai", "fake", "1", "1.0"}:
            return 1
        raise ValueError(
            f"Unknown label {raw_label!r} at row {csv_row}; expected real/nature/0 or ai/fake/1."
        )

    def _resolve_path(self, value) -> Path:
        p = Path(str(value).strip())
        if not p.is_absolute():
            p = self.csv_path.parent / p
        return p.resolve()

    def _path_key(self, value) -> str:
        return str(self._resolve_path(value)).casefold()
