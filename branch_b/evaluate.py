import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from branch_b.dataset import FrequencyDataset
from branch_b.frequency_model import build_frequency_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Branch B FFT-spectrum image classifier."
    )
    parser.add_argument("--csv", required=True, help="Evaluation CSV file")
    parser.add_argument(
        "--checkpoint",
        default="branch_b/models/frequency/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--exclude-generators",
        nargs="*",
        default=[],
        help="Optional list of generators to exclude from evaluation",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save metrics JSON",
    )
    parser.add_argument(
        "--output-predictions",
        default=None,
        help="Optional path to save per-image predictions CSV",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    return args


def evaluate_model(
    model,
    loader,
    criterion,
    device,
    checkpoint_path,
    checkpoint_epoch=None,
):
    loss_sum = 0.0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    progress = tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(
                device, dtype=torch.long, non_blocking=(device.type == "cuda")
            )
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            total += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predictions.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

            progress.set_postfix(
                loss=f"{loss_sum / total:.4f}",
                accuracy=f"{accuracy_score(all_labels, all_preds):.4f}",
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    avg_loss = loss_sum / total if total > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred))

    # Binary metrics (pos_label=1 for AI/fake)
    precision_ai = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall_ai = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_ai = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Metrics for Class 0 (nature/real)
    precision_real = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    recall_real = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Macro and weighted F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC (requires both classes present in y_true)
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None
    else:
        auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    class_counts = {
        "real/nature (0)": int((y_true == 0).sum()),
        "AI/fake (1)": int((y_true == 1).sum()),
    }

    results = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "samples": total,
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "class_counts": class_counts,
        "class_wise_results": {
            "real_nature_0": {
                "precision": precision_real,
                "recall": recall_real,
                "f1_score": f1_real,
                "support": class_counts["real/nature (0)"],
            },
            "ai_fake_1": {
                "precision": precision_ai,
                "recall": recall_ai,
                "f1_score": f1_ai,
                "support": class_counts["AI/fake (1)"],
            },
        },
    }
    return results, y_prob, y_pred


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Refusing to fall back to CPU; "
            "explicitly pass --device cpu to evaluate on CPU."
        )

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path.resolve()}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        checkpoint_epoch = checkpoint.get("epoch")
        if checkpoint.get("class_to_idx", {"real/nature": 0, "AI/fake": 1}) != {
            "real/nature": 0,
            "AI/fake": 1,
        }:
            raise ValueError(
                "Checkpoint class mapping is not real/nature=0, AI/fake=1."
            )
    else:
        model_state = checkpoint
        checkpoint_epoch = None
    if not isinstance(model_state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.resolve()}")

    dataset = FrequencyDataset(
        args.csv,
        image_col="path",
        exclude_generators=args.exclude_generators,
        deduplicate_paths=True,
    )
    if dataset.excluded_row_count:
        print(
            f"Excluded {dataset.excluded_row_count:,} rows with generators "
            f"{args.exclude_generators}.",
            flush=True,
        )
    if dataset.duplicate_row_count:
        print(
            f"Removed {dataset.duplicate_row_count:,} duplicate CSV rows "
            "for evaluation.",
            flush=True,
        )
    print(f"Total evaluation samples: {len(dataset):,}", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    model = build_frequency_model(pretrained=False).to(device)
    model.load_state_dict(model_state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    results, y_prob, y_pred = evaluate_model(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
    )

    print(json.dumps(results, indent=2), flush=True)

    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation metrics to: {out_json_path.resolve()}", flush=True)

    if args.output_predictions:
        out_pred_path = Path(args.output_predictions)
        out_pred_path.parent.mkdir(parents=True, exist_ok=True)
        df_preds = dataset.data.copy()
        df_preds["pred_label"] = y_pred
        df_preds["prob_ai"] = y_prob
        df_preds["prob_real"] = 1.0 - y_prob
        df_preds.to_csv(out_pred_path, index=False)
        print(f"Saved predictions to: {out_pred_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
