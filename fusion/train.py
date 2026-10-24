import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion.dataset import DualStreamDataset
from fusion.fusion_model import FeatureFusionModel, LateFusionModel

HELD_OUT_GENERATORS = ("imagenet_ai_0419_biggan",)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AI Image Detector Fusion Model.")
    parser.add_argument(
        "--train-csv",
        default="metadata/branch_b_splits/train.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--val-csv",
        default="metadata/branch_b_splits/val.csv",
        help="Path to validation CSV",
    )
    parser.add_argument(
        "--branch-a-checkpoint",
        default="checkpoints/imagenet_ai_0419_biggan/best_model.pt",
        help="Branch A spatial detector checkpoint",
    )
    parser.add_argument(
        "--branch-b-checkpoint",
        default="branch_b/models/frequency/best_model.pt",
        help="Branch B frequency detector checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for fusion head")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--output-dir",
        default="fusion/checkpoints",
        help="Directory to save fusion checkpoints",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def optimize_late_fusion_alpha(late_fusion_model, val_loader, device):
    """Find the optimal alpha weight on the validation set."""
    print("\n--- Optimizing Late Fusion Weight (Alpha) on Validation Set ---")
    all_probs_a = []
    all_probs_b = []
    all_labels = []

    with torch.no_grad():
        for spatial_imgs, freq_imgs, labels in tqdm(val_loader, desc="Collecting branch predictions"):
            _, p_a, p_b = late_fusion_model.predict_probs(spatial_imgs, freq_imgs)
            all_probs_a.extend(p_a.cpu().numpy().tolist())
            all_probs_b.extend(p_b.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    p_a = np.array(all_probs_a)
    p_b = np.array(all_probs_b)

    best_alpha = 0.5
    best_acc = 0.0

    # Grid search alpha in [0.0, 1.0]
    for alpha in np.linspace(0.0, 1.0, 101):
        p_fused = alpha * p_a + (1.0 - alpha) * p_b
        preds = (p_fused >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_alpha = float(alpha)

    # Evaluate individual branches on validation set
    acc_a = accuracy_score(y_true, (p_a >= 0.5).astype(int))
    acc_b = accuracy_score(y_true, (p_b >= 0.5).astype(int))
    f1_fused = f1_score(y_true, ((best_alpha * p_a + (1.0 - best_alpha) * p_b) >= 0.5).astype(int))
    auc_fused = roc_auc_score(y_true, best_alpha * p_a + (1.0 - best_alpha) * p_b)

    summary = {
        "best_alpha": round(best_alpha, 4),
        "val_accuracy_late_fusion": round(float(best_acc), 4),
        "val_f1_late_fusion": round(float(f1_fused), 4),
        "val_roc_auc_late_fusion": round(float(auc_fused), 4),
        "val_accuracy_branch_a": round(float(acc_a), 4),
        "val_accuracy_branch_b": round(float(acc_b), 4),
    }
    print(f"Branch A standalone validation accuracy: {acc_a:.4f}")
    print(f"Branch B standalone validation accuracy: {acc_b:.4f}")
    print(f"Optimal Alpha (weight on Branch A): {best_alpha:.2f}")
    print(f"Late Fusion validation accuracy: {best_acc:.4f} (F1: {f1_fused:.4f}, AUC: {auc_fused:.4f})")
    return summary


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Loading Dual-Stream Datasets ---")
    train_ds = DualStreamDataset(
        args.train_csv,
        exclude_generators=HELD_OUT_GENERATORS,
        deduplicate_paths=True,
    )
    val_ds = DualStreamDataset(
        args.val_csv,
        exclude_generators=HELD_OUT_GENERATORS,
        deduplicate_paths=True,
    )
    print(f"Train samples: {len(train_ds):,} (excluded BigGAN: {train_ds.excluded_row_count:,})")
    print(f"Val samples: {len(val_ds):,} (excluded BigGAN: {val_ds.excluded_row_count:,})")

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

    # 1. Late Fusion Optimization
    late_fusion = LateFusionModel(
        branch_a_checkpoint=args.branch_a_checkpoint,
        branch_b_checkpoint=args.branch_b_checkpoint,
        device=device,
    )
    late_fusion_summary = optimize_late_fusion_alpha(late_fusion, val_loader, device)

    # 2. Train Feature-Level Fusion Model
    print("\n--- Training Feature-Level Fusion Head ---")
    model = FeatureFusionModel(
        branch_a_checkpoint=args.branch_a_checkpoint,
        branch_b_checkpoint=args.branch_b_checkpoint,
        freeze_backbones=True,
    ).to(device)

    # Trainable parameters (only the fusion head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters in fusion head: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = 0.0
    history = []
    best_checkpoint_path = output_dir / "best_fusion_model.pt"
    last_checkpoint_path = output_dir / "last_fusion_model.pt"
    history_path = output_dir / "history.json"

    for epoch in range(args.epochs):
        epoch_start = time.monotonic()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", unit="batch")
        for spatial_imgs, freq_imgs, labels in pbar:
            spatial_imgs = spatial_imgs.to(device, non_blocking=(device.type == "cuda"))
            freq_imgs = freq_imgs.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, dtype=torch.long, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            outputs = model(spatial_imgs, freq_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss_sum += loss.item() * bs
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += bs
            pbar.set_postfix(
                loss=f"{train_loss_sum / train_total:.4f}",
                acc=f"{train_correct / train_total:.4f}",
            )

        scheduler.step()
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for spatial_imgs, freq_imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", unit="batch"):
                spatial_imgs = spatial_imgs.to(device, non_blocking=(device.type == "cuda"))
                freq_imgs = freq_imgs.to(device, non_blocking=(device.type == "cuda"))
                labels = labels.to(device, dtype=torch.long, non_blocking=(device.type == "cuda"))

                outputs = model(spatial_imgs, freq_imgs)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                bs = labels.size(0)
                val_loss_sum += loss.item() * bs
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += bs

                val_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
                val_targets.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_probs)
        epoch_sec = time.monotonic() - epoch_start

        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_f1": round(float(val_f1), 4),
            "val_roc_auc": round(float(val_auc), 4),
            "epoch_seconds": round(epoch_sec, 2),
        }
        history.append(record)
        print(f"\nEpoch {epoch + 1}: Val Acc = {val_acc:.4f}, Val F1 = {val_f1:.4f}, Val AUC = {val_auc:.4f}, Val Loss = {val_loss:.4f} ({epoch_sec:.1f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "fusion_head_state_dict": model.fusion_head.state_dict(),
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "val_f1": float(val_f1),
                    "val_roc_auc": float(val_auc),
                    "late_fusion_summary": late_fusion_summary,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )
            print(f"--> Saved new best fusion model to: {best_checkpoint_path}")

        torch.save(model.fusion_head.state_dict(), last_checkpoint_path)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_history": history,
                "late_fusion_summary": late_fusion_summary,
                "best_val_accuracy": best_val_acc,
            },
            f,
            indent=2,
        )

    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Saved artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
