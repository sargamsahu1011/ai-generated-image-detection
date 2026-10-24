"""
AI Image Detector (ResNet50, Leave-One-Generator-Out) - VS Code version
=======================================================================
Converted from the Google Colab notebook. Runs locally on Windows/Linux/Mac.

Folder layout expected (after downloading + unzipping tiny-genimage):

    <DATA_ROOT>/<generator>/<train|val>/<ai|nature>/<image files>

Usage (run in the VS Code terminal):

    python ai_image_detector.py audit                      # dataset + corruption + duplicate check
    python ai_image_detector.py prepare                    # build metadata.csv + LOGO split CSVs
    python ai_image_detector.py train    --held-out imagenet_ai_0419_biggan
    python ai_image_detector.py evaluate --held-out imagenet_ai_0419_biggan
    python ai_image_detector.py run      --held-out imagenet_ai_0419_biggan   # train + evaluate
    python ai_image_detector.py run      --held-out all                       # every generator

Optional flags: --data-root, --work-dir, --epochs (default 10), --batch-size, --workers
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE = 224
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3

# The duplicate pair discovered during the original audit.
DUPLICATE_TO_SKIP = {"imagenet_ai_0508_adm/train/nature/n02110185_129.JPEG"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
class Paths:
    def __init__(self, data_root, work_dir):
        self.raw = resolve_data_root(Path(data_root))
        self.work = Path(work_dir)
        self.metadata_dir = self.work / "metadata"
        self.metadata_csv = self.metadata_dir / "metadata.csv"
        self.split_dir = self.metadata_dir / "logo_splits"
        self.checkpoints = self.work / "checkpoints"
        self.results = self.work / "results"


def resolve_data_root(root: Path) -> Path:
    """If the zip extracted into one extra wrapper folder, descend into it."""
    if not root.exists():
        sys.exit(f"Data folder not found: {root}\nUse --data-root to point to the dataset.")
    for _ in range(3):
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        files = [p for p in root.iterdir() if p.is_file()]
        if len(subdirs) == 1 and not files:
            root = subdirs[0]
        else:
            break
    return root


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


# ----------------------------------------------------------------------
# 1. Audit (cells 4-10)
# ----------------------------------------------------------------------
def audit(paths: Paths):
    root = paths.raw
    print("=" * 70)
    print("DATASET AUDIT  ->", root)
    print("=" * 70)

    generators, splits, labels = Counter(), Counter(), Counter()
    combo = defaultdict(int)
    corrupt, formats, sizes = [], Counter(), Counter()
    hash_to_files = defaultdict(list)
    total = 0

    for path in tqdm(list(iter_images(root)), desc="Scanning images"):
        parts = path.relative_to(root).parts
        if len(parts) == 4:
            g, s, l, _ = parts
            generators[g] += 1
            splits[s] += 1
            labels[l] += 1
            combo[(g, s, l)] += 1
        total += 1

        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                formats[img.format] += 1
                sizes[img.size] += 1
        except Exception as e:
            corrupt.append((str(path), str(e)))

        with open(path, "rb") as f:
            hash_to_files[hashlib.md5(f.read()).hexdigest()].append(path)

    print("\nTotal images:", total)
    print("\nImages per generator:")
    for k, v in sorted(generators.items()):
        print(f"  {k}: {v}")
    print("\nImages per split:", dict(splits))
    print("Images per label:", dict(labels))

    print("\nGenerator x split x class:")
    for g in sorted(generators):
        for s in sorted(splits):
            row = " | ".join(f"{l}: {combo[(g, s, l)]}" for l in sorted(labels))
            print(f"  {g:32s} {s:5s} | {row}")

    print("\nCorrupt/unreadable:", len(corrupt))
    for p, e in corrupt[:10]:
        print("  ", p, e)

    print("\nFormats:", dict(formats))
    print("Most common sizes:", sizes.most_common(10))

    dups = {h: f for h, f in hash_to_files.items() if len(f) > 1}
    print("\nExact duplicate groups:", len(dups))
    for i, files in enumerate(list(dups.values())[:10], 1):
        print(f" Group {i}:")
        for f in files:
            print("   ", f)


# ----------------------------------------------------------------------
# 2. Metadata + LOGO splits (cells 11-13, 18-19 merged)
# ----------------------------------------------------------------------
def prepare(paths: Paths):
    paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    paths.split_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in tqdm(list(iter_images(paths.raw)), desc="Building metadata"):
        rel = path.relative_to(paths.raw)
        rel_posix = rel.as_posix()
        if rel_posix in DUPLICATE_TO_SKIP:
            continue
        if len(rel.parts) != 4:
            print("Unexpected path:", rel_posix)
            continue

        generator, split, label, _ = rel.parts
        if label == "nature":
            numeric_label = 0  # real
        elif label == "ai":
            numeric_label = 1  # fake
        else:
            print("Unexpected label:", label)
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                fmt = img.format
        except Exception as e:
            print("Could not read:", rel_posix, e)
            continue

        records.append({
            "path": str(path.resolve()),  # absolute path (fixes cells 18/19)
            "label": numeric_label,
            "label_name": label,
            "generator": generator,
            "split": split,
            "width": width,
            "height": height,
            "format": fmt,
        })

    df = pd.DataFrame(records)
    df.to_csv(paths.metadata_csv, index=False)
    print("\nMetadata saved:", paths.metadata_csv, "| records:", len(df))
    print(df["label_name"].value_counts().to_string())

    summary = []
    for held_out in sorted(df["generator"].unique()):
        seen = df[df["generator"] != held_out]
        train_df = seen[seen["split"] == "train"]
        val_df = seen[seen["split"] == "val"]
        test_df = df[df["generator"] == held_out]

        train_df.to_csv(paths.split_dir / f"{held_out}_train.csv", index=False)
        val_df.to_csv(paths.split_dir / f"{held_out}_val.csv", index=False)
        test_df.to_csv(paths.split_dir / f"{held_out}_test.csv", index=False)

        summary.append({
            "held_out_generator": held_out,
            "train_images": len(train_df),
            "validation_images": len(val_df),
            "test_images": len(test_df),
            "train_generators": sorted(train_df["generator"].unique().tolist()),
        })
        print(f"{held_out}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    with open(paths.split_dir / "logo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("LOGO splits saved in:", paths.split_dir)


# ----------------------------------------------------------------------
# 3. Dataset, transforms, loaders (cells 15-17)
# ----------------------------------------------------------------------
class AIGeneratedImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(int(row["label"]), dtype=torch.float32)


NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    NORM,
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORM,
])


def make_loader(df, transform, batch_size, workers, shuffle):
    return DataLoader(
        AIGeneratedImageDataset(df, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=USE_AMP,
    )


# ----------------------------------------------------------------------
# 4. Model (cell 22 / 28)
# ----------------------------------------------------------------------
def build_model(pretrained=True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, 1))
    return model.to(DEVICE)


# ----------------------------------------------------------------------
# 5. Train / validate (cell 26)
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, total, correct = 0.0, 0, 0

    bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in bar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).view(-1)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(images).view(-1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n = images.size(0)
        running_loss += loss.item() * n
        total += n
        correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss, total, correct = 0.0, 0, 0

    bar = tqdm(loader, desc="Validation", leave=False)
    for images, labels in bar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).view(-1)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(images).view(-1)
            loss = criterion(logits, labels)

        n = images.size(0)
        running_loss += loss.item() * n
        total += n
        correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


# ----------------------------------------------------------------------
# 6. Full training run for one held-out generator (cells 29 / 32)
# ----------------------------------------------------------------------
def load_split(paths, held_out):
    try:
        return (
            pd.read_csv(paths.split_dir / f"{held_out}_train.csv"),
            pd.read_csv(paths.split_dir / f"{held_out}_val.csv"),
            pd.read_csv(paths.split_dir / f"{held_out}_test.csv"),
        )
    except FileNotFoundError:
        sys.exit("Split CSVs not found. Run:  python ai_image_detector.py prepare")


def train(paths, held_out, epochs, batch_size, workers):
    train_df, val_df, test_df = load_split(paths, held_out)

    ckpt_dir = paths.checkpoints / held_out
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    train_loader = make_loader(train_df, train_transform, batch_size, workers, True)
    val_loader = make_loader(val_df, eval_transform, batch_size, workers, False)

    model = build_model(pretrained=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    print("=" * 70)
    print("TRAINING | held-out generator:", held_out)
    print(f"Device: {DEVICE} | AMP: {USE_AMP} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Unseen test: {len(test_df)}")
    print("=" * 70)

    best_val_loss, best_epoch, patience, history = float("inf"), 0, 0, []

    for epoch in range(1, epochs + 1):
        start = time.time()
        print(f"\nEPOCH {epoch}/{epochs}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, va_acc = validate(model, val_loader, criterion)
        elapsed = time.time() - start

        print(f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"Val loss {va_loss:.4f} acc {va_acc:.4f} | {elapsed / 60:.2f} min")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_accuracy": tr_acc,
            "val_loss": va_loss, "val_accuracy": va_acc,
            "epoch_time_seconds": elapsed,
        })

        if va_loss < best_val_loss:
            best_val_loss, best_epoch, patience = va_loss, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": va_loss,
                "val_accuracy": va_acc,
                "held_out_generator": held_out,
            }, ckpt_path)
            print("  New best model saved")
        else:
            patience += 1
            print(f"  No improvement ({patience}/{EARLY_STOPPING_PATIENCE})")
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTRAINING COMPLETE | best epoch:", best_epoch,
          "| best val loss:", round(best_val_loss, 4))
    print("Checkpoint:", ckpt_path)


# ----------------------------------------------------------------------
# 7. Evaluation on the unseen generator (cell 30)
# ----------------------------------------------------------------------
def evaluate(paths, held_out, batch_size):
    ckpt_path = paths.checkpoints / held_out / "best_model.pt"
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}\nTrain first with the 'train' command.")

    _, _, test_df = load_split(paths, held_out)
    results_dir = paths.results / held_out
    results_dir.mkdir(parents=True, exist_ok=True)

    # num_workers=0 keeps evaluation simple and Windows-safe
    test_loader = make_loader(test_df, eval_transform, batch_size, 0, False)

    model = build_model(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 70)
    print("EVALUATION | unseen generator:", held_out)
    print("Checkpoint epoch:", ckpt["epoch"], "| val loss:", round(ckpt["val_loss"], 4))
    print("=" * 70)

    labels_all, probs_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {held_out}"):
            images = images.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(images).view(-1)
            probs_all.extend(torch.sigmoid(logits).float().cpu().numpy())
            labels_all.extend(labels.numpy())

    y_true = np.array(labels_all).astype(int)
    y_prob = np.array(probs_all)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"PR-AUC   : {pr:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\n", classification_report(y_true, y_pred, target_names=["Real", "AI"],
                                      digits=4, zero_division=0))

    preds = test_df.copy()
    preds["probability_ai"] = y_prob
    preds["prediction"] = y_pred
    preds["prediction_name"] = np.where(y_pred == 1, "ai", "real")
    preds.to_csv(results_dir / "test_predictions.csv", index=False)

    metrics = {
        "experiment": "Branch A Spatial ResNet50",
        "held_out_generator": held_out,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "validation_loss": float(ckpt["val_loss"]),
        "validation_accuracy": float(ckpt["val_accuracy"]),
        "test_samples": int(len(y_true)),
        "threshold": 0.5,
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "roc_auc": float(roc), "pr_auc": float(pr),
        "confusion_matrix": cm.tolist(),
    }
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results in:", results_dir)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI Image Detector (ResNet50, LOGO)")
    parser.add_argument("command", choices=["audit", "prepare", "train", "evaluate", "run"])
    parser.add_argument("--data-root", default="data/raw",
                        help="Folder containing the generator folders (default: data/raw)")
    parser.add_argument("--work-dir", default="outputs",
                        help="Where metadata, checkpoints and results go (default: outputs)")
    parser.add_argument("--held-out", default="imagenet_ai_0419_biggan",
                        help="Generator to hold out for testing, or 'all'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2,
                        help="DataLoader workers (default 0 on Windows, 2 elsewhere)")
    args = parser.parse_args()

    print("PyTorch:", torch.__version__, "| Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU found - training ResNet50 on CPU will be very slow.")

    paths = Paths(args.data_root, args.work_dir)

    if args.command == "audit":
        audit(paths)
        return
    if args.command == "prepare":
        prepare(paths)
        return

    if args.held_out == "all":
        df = pd.read_csv(paths.metadata_csv)
        targets = sorted(df["generator"].unique())
    else:
        targets = [args.held_out]

    for g in targets:
        if args.command in ("train", "run"):
            train(paths, g, args.epochs, args.batch_size, args.workers)
        if args.command in ("evaluate", "run"):
            evaluate(paths, g, args.batch_size)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
