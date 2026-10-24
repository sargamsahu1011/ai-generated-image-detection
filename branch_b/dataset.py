from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from branch_b.fft_utils import image_to_fft


class FrequencyDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_col="path",
        label_col="label",
        image_size=224,
        generator_col="generator",
        exclude_generators=(),
        deduplicate_paths=False,
    ):
        self.csv_path = Path(csv_path).resolve()
        self.data = pd.read_csv(self.csv_path)
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = image_size

        missing = {
            image_col,
            label_col,
        } - set(self.data.columns)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}. "
                f"Available columns: {list(self.data.columns)}"
            )

        self.data = self.data.copy()
        self.data["_frequency_label"] = [
            self._map_label(value, row_index + 2)
            for row_index, value in enumerate(self.data[label_col])
        ]

        excluded = {str(name).strip().lower() for name in exclude_generators}
        self.excluded_row_count = 0
        if excluded:
            if generator_col not in self.data.columns:
                raise ValueError(
                    f"Cannot exclude generators because CSV column "
                    f"{generator_col!r} is missing."
                )
            generators = self.data[generator_col].astype(str).str.strip().str.lower()
            excluded_rows = generators.isin(excluded)
            self.excluded_row_count = int(excluded_rows.sum())
            self.data = self.data.loc[~excluded_rows].copy()
        self.generator_col = generator_col

        empty_paths = self.data[image_col].isna() | (
            self.data[image_col].astype(str).str.strip() == ""
        )
        if empty_paths.any():
            row_index = int(empty_paths[empty_paths].index[0]) + 2
            raise ValueError(f"Empty image path in {self.csv_path}, CSV row {row_index}.")

        self.data["_frequency_path_key"] = self.data[image_col].map(
            self._path_key
        )
        self.duplicate_row_count = 0
        if deduplicate_paths:
            consistency_columns = [
                column
                for column in (label_col, generator_col, "label_name", "split")
                if column in self.data.columns
            ]
            conflicts = (
                self.data.groupby("_frequency_path_key", sort=False)[
                    consistency_columns
                ]
                .nunique(dropna=False)
                .gt(1)
                .any(axis=1)
            )
            if conflicts.any():
                conflicting_path = conflicts[conflicts].index[0]
                raise ValueError(
                    f"Duplicate image path has conflicting metadata in "
                    f"{self.csv_path}: {conflicting_path}"
                )
            deduplicated = self.data.drop_duplicates(
                subset="_frequency_path_key",
                keep="first",
            )
            self.duplicate_row_count = len(self.data) - len(deduplicated)
            self.data = deduplicated.copy()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = self._resolve_path(row[self.image_col])
        csv_row = int(row.name) + 2
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            image = image_to_fft(image, size=self.image_size)
            image = self.transform(image)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to read or transform image {image_path} "
                f"from {self.csv_path}, CSV row {csv_row}: {exc}"
            ) from exc

        return image, int(row["_frequency_label"])

    def _map_label(self, raw_label, csv_row):
        label = str(raw_label).strip().lower()
        if label in {"real", "nature", "0", "0.0"}:
            return 0
        if label in {"ai", "fake", "1", "1.0"}:
            return 1
        else:
            raise ValueError(
                f"Unknown label {raw_label!r} at CSV row {csv_row} "
                f"in {self.csv_path}; expected real/nature/0 or AI/fake/1."
            )

    def _resolve_path(self, value):
        image_path = Path(str(value).strip())
        if not image_path.is_absolute():
            image_path = self.csv_path.parent / image_path
        return image_path.resolve()

    def _path_key(self, value):
        return str(self._resolve_path(value)).casefold()