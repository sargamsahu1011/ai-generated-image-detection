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

from fusion.dataset import DualStreamDataset
from fusion.fusion_model import FeatureFusionModel, LateFusionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AI Image Detector Fusion Models.")
    parser.add_argument("--csv", required=True, help="Evaluation CSV")
    parser.add_argument(
        "--fusion-checkpoint",
        default="fusion/checkpoints/best_fusion_model.pt",
        help="Path to feature fusion model checkpoint",
    )
    parser.add_argument(
        "--branch-a-checkpoint",
        default="checkpoints/imagenet_ai_0419_biggan/best_model.pt",
        help="Branch A checkpoint",
    )
    parser.add_argument(
        "--branch-b-checkpoint",
        default="branch_b/models/frequency/best_model.pt",
        help="Branch B checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None, help="Late fusion alpha (default: loaded from checkpoint)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--exclude-generators", nargs="*", default=[])
    parser.add_argument("--output-json", default=None, help="Path to save metrics JSON")
    parser.add_argument("--output-predictions", default=None, help="Path to save predictions CSV")
    return parser.parse_args()


def calculate_metrics(y_true, y_pred, y_prob):
    acc = float(accuracy_score(y_true, y_pred))
    prec_ai = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_ai = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_ai = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    prec_real = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_real = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": cm,
        "class_wise": {
            "real_nature_0": {
                "precision": round(prec_real, 4),
                "recall": round(rec_real, 4),
                "f1_score": round(f1_real, 4),
                "support": int((y_true == 0).sum()),
            },
            "ai_fake_1": {
                "precision": round(prec_ai, 4),
                "recall": round(rec_ai, 4),
                "f1_score": round(f1_ai, 4),
                "support": int((y_true == 1).sum()),
            },
        },
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    dataset = DualStreamDataset(
        args.csv,
        exclude_generators=args.exclude_generators,
        deduplicate_paths=True,
    )
    print(f"Loaded {len(dataset):,} samples from {args.csv}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # 1. Load Late Fusion Models
    late_fusion = LateFusionModel(
        branch_a_checkpoint=args.branch_a_checkpoint,
        branch_b_checkpoint=args.branch_b_checkpoint,
        device=device,
    )

    # Load alpha from fusion checkpoint if available
    alpha = args.alpha
    if alpha is None:
        try:
            ckpt = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
            if "late_fusion_summary" in ckpt and "best_alpha" in ckpt["late_fusion_summary"]:
                alpha = float(ckpt["late_fusion_summary"]["best_alpha"])
                print(f"Using saved optimal alpha from checkpoint: {alpha:.2f}")
        except Exception:
            alpha = 0.5
            print(f"Defaulting alpha to {alpha}")

    # 2. Load Feature-Level Fusion Model
    feature_fusion_available = False
    feat_model = None
    if Path(args.fusion_checkpoint).is_file():
        try:
            feat_model = FeatureFusionModel(
                branch_a_checkpoint=args.branch_a_checkpoint,
                branch_b_checkpoint=args.branch_b_checkpoint,
                freeze_backbones=True,
            ).to(device)
            ckpt = torch.load(args.fusion_checkpoint, map_location=device, weights_only=False)
            head_state = ckpt["fusion_head_state_dict"] if "fusion_head_state_dict" in ckpt else ckpt
            feat_model.fusion_head.load_state_dict(head_state)
            feat_model.eval()
            feature_fusion_available = True
            print(f"Loaded FeatureFusionModel from {args.fusion_checkpoint}")
        except Exception as e:
            print(f"Could not load FeatureFusionModel: {e}")

    # Evaluate across all streams
    all_y = []
    probs_a = []
    probs_b = []
    probs_late = []
    probs_feat = []

    with torch.no_grad():
        for spatial_imgs, freq_imgs, labels in tqdm(loader, desc="Evaluating All Streams"):
            spatial_imgs = spatial_imgs.to(device)
            freq_imgs = freq_imgs.to(device)

            # Individual and Late Fusion
            p_late, p_a, p_b = late_fusion.predict_probs(spatial_imgs, freq_imgs, alpha=alpha)

            probs_a.extend(p_a.cpu().numpy().tolist())
            probs_b.extend(p_b.cpu().numpy().tolist())
            probs_late.extend(p_late.cpu().numpy().tolist())
            all_y.extend(labels.numpy().tolist())

            # Feature Fusion
            if feature_fusion_available:
                logits_feat = feat_model(spatial_imgs, freq_imgs)
                p_feat = torch.softmax(logits_feat, dim=1)[:, 1]
                probs_feat.extend(p_feat.cpu().numpy().tolist())

    y_true = np.array(all_y)
    p_a = np.array(probs_a)
    p_b = np.array(probs_b)
    p_late = np.array(probs_late)

    metrics_a = calculate_metrics(y_true, (p_a >= 0.5).astype(int), p_a)
    metrics_b = calculate_metrics(y_true, (p_b >= 0.5).astype(int), p_b)
    metrics_late = calculate_metrics(y_true, (p_late >= 0.5).astype(int), p_late)

    results = {
        "dataset_csv": str(Path(args.csv).resolve()),
        "total_samples": len(y_true),
        "branch_a_spatial": metrics_a,
        "branch_b_frequency": metrics_b,
        "late_fusion": {
            "alpha_weight_branch_a": alpha,
            **metrics_late,
        },
    }

    if feature_fusion_available:
        p_feat = np.array(probs_feat)
        metrics_feat = calculate_metrics(y_true, (p_feat >= 0.5).astype(int), p_feat)
        results["feature_fusion"] = metrics_feat

    print("\n" + "=" * 75)
    print(f"EVALUATION RESULTS ({Path(args.csv).name} - {len(y_true):,} samples)")
    print("=" * 75)
    print(f"{'Method':<26} | {'Accuracy':<9} | {'Macro F1':<9} | {'ROC-AUC':<9}")
    print("-" * 75)
    print(f"{'Branch A (Spatial ResNet50)':<26} | {metrics_a['accuracy']:<9.4f} | {metrics_a['macro_f1']:<9.4f} | {metrics_a['roc_auc']:<9.4f}")
    print(f"{'Branch B (Frequency ResNet50)':<26} | {metrics_b['accuracy']:<9.4f} | {metrics_b['macro_f1']:<9.4f} | {metrics_b['roc_auc']:<9.4f}")
    print(f"{'Late Fusion (Ensemble)':<26} | {metrics_late['accuracy']:<9.4f} | {metrics_late['macro_f1']:<9.4f} | {metrics_late['roc_auc']:<9.4f}")
    if feature_fusion_available:
        print(f"{'Feature Fusion (Concat+MLP)':<26} | {metrics_feat['accuracy']:<9.4f} | {metrics_feat['macro_f1']:<9.4f} | {metrics_feat['roc_auc']:<9.4f}")
    print("=" * 75)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation results to {out_json.resolve()}")

    if args.output_predictions:
        out_csv = Path(args.output_predictions)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out = dataset.data.copy()
        df_out["prob_ai_branch_a"] = p_a
        df_out["prob_ai_branch_b"] = p_b
        df_out["prob_ai_late_fusion"] = p_late
        df_out["pred_late_fusion"] = (p_late >= 0.5).astype(int)
        if feature_fusion_available:
            df_out["prob_ai_feature_fusion"] = p_feat
            df_out["pred_feature_fusion"] = (p_feat >= 0.5).astype(int)
        df_out.to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv.resolve()}")


if __name__ == "__main__":
    main()
