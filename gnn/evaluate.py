#!/usr/bin/env python3
"""
Evaluation script for PackHero GNN packer classifier.

Usage:
    python -m gnn.evaluate --checkpoint checkpoints/best_model.pt --data-root packhero-dataset

Features:
- Load trained model from checkpoint
- Compute metrics: accuracy, precision, recall, F1-score
- Generate confusion matrix
- Per-class performance breakdown
- Sample-level predictions export
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.model import create_model, PackerGNN, PackerGMN
from gnn.dataset import get_dataloaders, PackerDataset, PACKER_LABELS, BINARY_LABELS
from gnn.utils import (
    set_seed,
    get_device,
    compute_metrics,
    get_classification_report,
    plot_confusion_matrix,
    load_checkpoint,
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained GNN model
        loader: Data loader for evaluation
        device: Device to evaluate on
        label_names: List of class names

    Returns:
        Dictionary containing all evaluation results
    """
    model.eval()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)

        # Forward pass
        logits = model.forward_batch(batch)
        loss = criterion(logits, batch.y)

        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        # Collect results
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        # Collect filenames if available
        if hasattr(batch, 'filename'):
            if isinstance(batch.filename, list):
                all_filenames.extend(batch.filename)
            else:
                all_filenames.extend([batch.filename])

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Compute overall metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics = compute_metrics(y_true, y_pred, average="weighted")

    # Per-class metrics
    per_class_metrics = {}
    unique_labels = np.unique(y_true)

    for label_idx in unique_labels:
        mask = y_true == label_idx
        label_name = label_names[label_idx] if label_names and label_idx < len(label_names) else str(label_idx)

        # Binary metrics for this class
        y_true_binary = (y_true == label_idx).astype(int)
        y_pred_binary = (y_pred == label_idx).astype(int)

        class_metrics = compute_metrics(y_true_binary, y_pred_binary, average="binary")
        class_metrics["support"] = int(mask.sum())

        per_class_metrics[label_name] = class_metrics

    # Compile results
    results = {
        "loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "total_samples": total_samples,
        "per_class": per_class_metrics,
        "predictions": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_probs": y_probs.tolist(),
            "filenames": all_filenames,
        },
    }

    return results


def print_evaluation_report(
    results: Dict,
    label_names: Optional[List[str]] = None,
) -> None:
    """
    Print a formatted evaluation report.

    Args:
        results: Evaluation results dictionary
        label_names: List of class names
    """
    print("\n" + "=" * 70)
    print("                    EVALUATION REPORT")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  {'Loss:':<20} {results['loss']:.4f}")
    print(f"  {'Accuracy:':<20} {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  {'Precision:':<20} {results['precision']:.4f}")
    print(f"  {'Recall:':<20} {results['recall']:.4f}")
    print(f"  {'F1 Score:':<20} {results['f1']:.4f}")
    print(f"  {'Total Samples:':<20} {results['total_samples']}")

    print(f"\nPer-Class Performance:")
    print("-" * 70)
    print(f"  {'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)

    for class_name, class_metrics in results["per_class"].items():
        print(
            f"  {class_name:<20} "
            f"{class_metrics['precision']:<12.4f} "
            f"{class_metrics['recall']:<12.4f} "
            f"{class_metrics['f1']:<12.4f} "
            f"{class_metrics['support']:<10}"
        )

    print("-" * 70)

    # Print classification report
    y_true = np.array(results["predictions"]["y_true"])
    y_pred = np.array(results["predictions"]["y_pred"])

    print(f"\nDetailed Classification Report:")
    print(get_classification_report(y_true, y_pred, label_names=label_names))


def analyze_errors(
    results: Dict,
    label_names: Optional[List[str]] = None,
    max_errors: int = 10,
) -> None:
    """
    Analyze and print misclassified samples.

    Args:
        results: Evaluation results dictionary
        label_names: List of class names
        max_errors: Maximum number of errors to display
    """
    y_true = np.array(results["predictions"]["y_true"])
    y_pred = np.array(results["predictions"]["y_pred"])
    y_probs = np.array(results["predictions"]["y_probs"])
    filenames = results["predictions"]["filenames"]

    # Find misclassified samples
    errors_mask = y_true != y_pred
    error_indices = np.where(errors_mask)[0]

    if len(error_indices) == 0:
        print("\nNo misclassified samples!")
        return

    print(f"\n" + "=" * 70)
    print(f"ERROR ANALYSIS ({len(error_indices)} misclassified samples)")
    print("=" * 70)

    # Sort by confidence (highest confidence errors first)
    error_confidences = []
    for idx in error_indices:
        pred_class = y_pred[idx]
        confidence = y_probs[idx][pred_class]
        error_confidences.append((idx, confidence))

    error_confidences.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {min(max_errors, len(error_confidences))} High-Confidence Errors:")
    print("-" * 70)

    for i, (idx, confidence) in enumerate(error_confidences[:max_errors]):
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        true_name = label_names[true_label] if label_names else str(true_label)
        pred_name = label_names[pred_label] if label_names else str(pred_label)

        filename = filenames[idx] if idx < len(filenames) else "N/A"

        print(f"\n  [{i+1}] File: {filename}")
        print(f"      True: {true_name} -> Predicted: {pred_name}")
        print(f"      Confidence: {confidence:.4f}")

        # Show probability distribution
        if label_names:
            print(f"      Probabilities:")
            for j, name in enumerate(label_names):
                print(f"        {name}: {y_probs[idx][j]:.4f}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate PackHero GNN packer classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )

    # Data arguments
    parser.add_argument(
        "--data-root",
        type=str,
        default="packhero-dataset",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        default=None,
        help="Directory containing .pt graph files",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Classification task (auto-detected from checkpoint if not specified)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which data split to evaluate on",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample predictions to JSON",
    )
    parser.add_argument(
        "--save-confusion-matrix",
        action="store_true",
        help="Save confusion matrix plot",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show analysis of misclassified samples",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum number of errors to display",
    )

    # Other arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get("config", {})

    # Determine task
    task = args.task or config.get("task", "multiclass")
    print(f"Task: {task}")

    # Create data loaders
    print(f"Loading dataset from: {args.data_root}")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        root=args.data_root,
        graph_dir=args.graph_dir,
        task=task,
        batch_size=args.batch_size,
        random_state=args.seed,
    )

    # Select the appropriate loader
    if args.split == "train":
        loader = train_loader
        split_name = "Training Set"
    elif args.split == "val":
        loader = val_loader
        split_name = "Validation Set"
    elif args.split == "test":
        loader = test_loader
        split_name = "Test Set"
    else:
        # Evaluate on all data
        from torch.utils.data import ConcatDataset
        all_data = ConcatDataset([
            train_loader.dataset,
            val_loader.dataset,
            test_loader.dataset
        ])
        loader = DataLoader(all_data, batch_size=args.batch_size)
        split_name = "All Data"

    print(f"Evaluating on: {split_name}")

    # Create model
    model_type = config.get("model_type", "gcn")
    num_classes = config.get("num_classes", dataset.num_classes)
    in_channels = config.get("in_channels", 5)
    hidden_channels = config.get("hidden_channels", 64)
    num_layers = config.get("num_layers", 3)
    dropout = config.get("dropout", 0.5)
    heads = config.get("heads", 4)

    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        heads=heads,
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded: {model_type.upper()} with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Get label names
    label_names = config.get("label_names", dataset.label_names)

    # Evaluate
    print(f"\nRunning evaluation...")
    results = evaluate_model(model, loader, device, label_names)

    # Print report
    print_evaluation_report(results, label_names)

    # Error analysis
    if args.show_errors:
        analyze_errors(results, label_names, args.max_errors)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    if args.save_predictions:
        predictions_path = output_dir / f"predictions_{args.split}.json"
        with open(predictions_path, "w") as f:
            # Create a cleaner format
            pred_data = {
                "split": args.split,
                "samples": [],
            }
            y_true = results["predictions"]["y_true"]
            y_pred = results["predictions"]["y_pred"]
            y_probs = results["predictions"]["y_probs"]
            filenames = results["predictions"]["filenames"]

            for i in range(len(y_true)):
                sample = {
                    "filename": filenames[i] if i < len(filenames) else f"sample_{i}",
                    "true_label": label_names[y_true[i]] if label_names else y_true[i],
                    "predicted_label": label_names[y_pred[i]] if label_names else y_pred[i],
                    "correct": y_true[i] == y_pred[i],
                    "probabilities": {
                        label_names[j] if label_names else str(j): float(y_probs[i][j])
                        for j in range(len(y_probs[i]))
                    },
                }
                pred_data["samples"].append(sample)

            json.dump(pred_data, f, indent=2)
        print(f"\nPredictions saved to: {predictions_path}")

    # Save confusion matrix
    if args.save_confusion_matrix:
        cm_path = output_dir / f"confusion_matrix_{args.split}.png"
        y_true = np.array(results["predictions"]["y_true"])
        y_pred = np.array(results["predictions"]["y_pred"])
        plot_confusion_matrix(
            y_true, y_pred,
            label_names=label_names,
            save_path=str(cm_path),
        )

    # Save summary results
    summary_path = output_dir / f"evaluation_summary_{args.split}.json"
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "model_type": model_type,
        "task": task,
        "loss": results["loss"],
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "total_samples": results["total_samples"],
        "per_class": results["per_class"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
