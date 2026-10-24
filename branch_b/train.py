import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from branch_b.dataset import FrequencyDataset
from branch_b.frequency_model import build_frequency_model


HELD_OUT_GENERATORS = {"imagenet_ai_0419_biggan"}
EXPECTED_GENERATORS = {
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0508_adm",
    "imagenet_glide",
    "imagenet_midjourney",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Branch B FFT-spectrum image classifier."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", default="branch_b/models/frequency")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be greater than 0")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    return args


def validate_datasets(train_ds, val_ds):
    datasets = (("train", train_ds, "train"), ("validation", val_ds, "val"))
    generator_sets = []

    for name, dataset, expected_split in datasets:
        frame = dataset.data
        if "generator" not in frame.columns or "split" not in frame.columns:
            raise ValueError(
                f"The {name} CSV must contain 'generator' and 'split' columns."
            )

        split_values = set(frame["split"].astype(str).str.strip().str.lower())
        if split_values != {expected_split}:
            raise ValueError(
                f"The {name} CSV must contain only split={expected_split!r}; "
                f"found {sorted(split_values)}."
            )

        generators = set(frame["generator"].astype(str).str.strip().str.lower())
        if generators != EXPECTED_GENERATORS:
            raise ValueError(
                f"The {name} CSV must contain exactly the six prepared "
                f"non-held-out generators; found {sorted(generators)}."
            )
        if generators & HELD_OUT_GENERATORS:
            raise ValueError(f"The held-out BigGAN generator remains in {name}.")
        generator_sets.append(generators)

        labels = set(dataset.data["_frequency_label"].unique())
        if labels != {0, 1}:
            raise ValueError(
                f"The {name} dataset must contain both labels 0 and 1; "
                f"found {sorted(labels)}."
            )

    if generator_sets[0] != generator_sets[1]:
        raise ValueError("Train and validation generator sets do not match.")

    train_paths = set(train_ds.data["_frequency_path_key"])
    val_paths = set(val_ds.data["_frequency_path_key"])
    overlap = train_paths & val_paths
    if overlap:
        raise ValueError(
            f"Found {len(overlap)} image paths in both train and validation."
        )


def save_history(history, output_path):
    with output_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Refusing to fall back to CPU; "
            "verify the PyTorch CUDA installation or explicitly pass --device cpu."
        )

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        print(
            "CUDA memory: "
            f"{free_memory / 1024**3:.2f} GiB free / "
            f"{total_memory / 1024**3:.2f} GiB total",
            flush=True,
        )

    train_ds = FrequencyDataset(
        args.train_csv,
        image_col="path",
        exclude_generators=HELD_OUT_GENERATORS,
        deduplicate_paths=True,
    )
    val_ds = FrequencyDataset(
        args.val_csv,
        image_col="path",
        exclude_generators=HELD_OUT_GENERATORS,
        deduplicate_paths=True,
    )
    if train_ds.excluded_row_count or val_ds.excluded_row_count:
        print(
            "Excluded held-out generator rows without changing either CSV: "
            f"{train_ds.excluded_row_count} train, "
            f"{val_ds.excluded_row_count} validation.",
            flush=True,
        )
    validate_datasets(train_ds, val_ds)
    print(
        f"Removed duplicate rows without changing either CSV: "
        f"{train_ds.duplicate_row_count:,} train, "
        f"{val_ds.duplicate_row_count:,} validation.",
        flush=True,
    )
    print(
        f"Using {len(train_ds):,} unique training and {len(val_ds):,} unique "
        f"validation images from {len(EXPECTED_GENERATORS)} non-held-out generators.",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = (
        output_dir / "best_model.pt",
        output_dir / "last_model_state.pt",
        output_dir / "history.json",
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        names = ", ".join(str(path.resolve()) for path in existing_outputs)
        raise FileExistsError(
            f"Refusing to overwrite existing experiment outputs: {names}. "
            "Choose a new --output-dir or preserve these files before retrying."
        )

    if args.pretrained:
        print(
            "Loading pretrained ResNet50 weights; torchvision will show download "
            "progress if the weights are not cached.",
            flush=True,
        )
    try:
        model = build_frequency_model(pretrained=args.pretrained).to(device)
    except Exception as exc:
        if args.pretrained:
            raise RuntimeError(
                "Could not initialize ResNet50 with pretrained weights. "
                "Check the download output/network and retry when weights are "
                "available."
            ) from exc
        raise

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_val_accuracy = -1.0
    history = []
    run_started = time.monotonic()
    batches_per_epoch = len(train_loader)
    print(
        f"Starting {args.epochs} epochs, batch size {args.batch_size}, "
        f"{batches_per_epoch:,} training batches/epoch.",
        flush=True,
    )

    for epoch in range(args.epochs):
        epoch_started = time.monotonic()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} train",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, (images, labels) in enumerate(train_progress):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(
                device, dtype=torch.long, non_blocking=(device.type == "cuda")
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += batch_size
            train_progress.set_postfix(
                loss=f"{train_loss_sum / train_total:.4f}",
                accuracy=f"{train_correct / train_total:.4f}",
                elapsed=f"{time.monotonic() - run_started:.0f}s",
            )

            if epoch == 0 and batch_index == min(2, batches_per_epoch - 1):
                measured_batches = batch_index + 1
                measured_seconds = time.monotonic() - epoch_started
                estimate = (
                    measured_seconds / measured_batches
                    * batches_per_epoch
                    * args.epochs
                )
                print(
                    "\nInitial training-time estimate from "
                    f"{measured_batches} batch(es): "
                    f"{estimate / 3600:.2f} hours, excluding validation; "
                    "this estimate will be refined by actual epoch timings.",
                    flush=True,
                )

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} validation",
            unit="batch",
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for images, labels in val_progress:
                images = images.to(device, non_blocking=(device.type == "cuda"))
                labels = labels.to(
                    device, dtype=torch.long, non_blocking=(device.type == "cuda")
                )

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += batch_size
                val_progress.set_postfix(
                    loss=f"{val_loss_sum / val_total:.4f}",
                    accuracy=f"{val_correct / val_total:.4f}",
                    elapsed=f"{time.monotonic() - run_started:.0f}s",
                )

        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total
        epoch_seconds = time.monotonic() - epoch_started
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": time.monotonic() - run_started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "class_to_idx": {"real/nature": 0, "AI/fake": 1},
                    "args": vars(args),
                },
                output_paths[0],
            )
        torch.save(model.state_dict(), output_paths[1])
        save_history(history, output_paths[2])

    total_seconds = time.monotonic() - run_started
    print(f"Best validation accuracy: {best_val_accuracy:.4f}", flush=True)
    print(f"Total training runtime: {total_seconds:.1f}s", flush=True)
    print(f"Saved outputs to: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
