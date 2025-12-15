#!/usr/bin/env python3
"""
Training script for PackHero GNN packer classifier.

Usage:
    python -m gnn.train --data-root packhero-dataset --epochs 100 --model gcn

Features:
- Training with early stopping
- Validation monitoring
- Checkpoint saving
- Training history logging
- Multiple model architectures (GCN, GAT, SAGE, GIN, GMN)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from torch_geometric.loader import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.model import create_model, PackerGNNConfig
from gnn.dataset import get_dataloaders, PackerDataset
from gnn.utils import (
    set_seed,
    get_device,
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
    plot_training_history,
    EarlyStopping,
    count_parameters,
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: The GNN model
        loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        criterion: Loss function

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model.forward_batch(batch)
        loss = criterion(out, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluate the model.

    Args:
        model: The GNN model
        loader: Data loader
        device: Device to evaluate on
        criterion: Loss function

    Returns:
        Tuple of (average loss, accuracy, all predictions, all labels)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)

        # Forward pass
        out = model.forward_batch(batch)
        loss = criterion(out, batch.y)

        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy, all_preds, all_labels


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict,
) -> Dict[str, List[float]]:
    """
    Full training loop with validation.

    Args:
        model: The GNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        config: Training configuration

    Returns:
        Training history dictionary
    """
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if config.get("optimizer", "adam") == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.01),
        )
    else:
        optimizer = Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )

    # Learning rate scheduler
    if config.get("scheduler", "plateau") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get("patience", 15),
        min_delta=config.get("min_delta", 0.001),
        mode="min",
        verbose=True,
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Checkpoint directory
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Model parameters: {count_parameters(model):,}")
    print("-" * 60)

    for epoch in range(1, config["epochs"] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, criterion
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)

        # Update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Print progress
        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                path=str(checkpoint_dir / "best_model.pt"),
                val_acc=val_acc,
                config=config,
            )

        # Check early stopping
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {early_stopping.best_epoch}")
            break

    print("-" * 60)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=val_loss,
        path=str(checkpoint_dir / "final_model.pt"),
        val_acc=val_acc,
        config=config,
    )

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Plot training curves
    try:
        fig = plot_training_history(
            history,
            save_path=str(checkpoint_dir / "training_curves.png")
        )
    except Exception as e:
        print(f"Could not save training curves: {e}")

    return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train PackHero GNN packer classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default="multiclass",
        choices=["multiclass", "binary"],
        help="Classification task",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["gcn", "gat", "sage", "gin", "gmn"],
        help="GNN model architecture",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads (for GAT)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help="Optimizer",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine"],
        help="Learning rate scheduler",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers",
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Create data loaders
    print(f"\nLoading dataset from: {args.data_root}")
    try:
        train_loader, val_loader, test_loader, dataset = get_dataloaders(
            root=args.data_root,
            graph_dir=args.graph_dir,
            task=args.task,
            batch_size=args.batch_size,
            random_state=args.seed,
            num_workers=args.num_workers,
        )
    except ValueError as e:
        print(f"\nError loading dataset: {e}")
        print("\nTo train the model, you need to:")
        print("1. Extract call graphs from PE files using radare2")
        print("2. Convert graphs to PyTorch Geometric format (.pt files)")
        print("3. Place .pt files in the 'graphs' subdirectory of your dataset")
        print("\nAlternatively, use the --demo flag to run with synthetic data.")
        sys.exit(1)

    # Determine number of classes and features
    num_classes = dataset.num_classes
    in_channels = 5  # Default PackHero features

    # Try to get actual feature dimension from first sample
    if len(dataset) > 0:
        sample = dataset[0]
        if sample.x is not None:
            in_channels = sample.x.shape[1]

    print(f"Task: {args.task}")
    print(f"Number of classes: {num_classes}")
    print(f"Input features: {in_channels}")
    print(f"Label names: {dataset.label_names}")

    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    model = create_model(
        model_type=args.model,
        num_classes=num_classes,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
    )

    # Training configuration
    config = {
        "model_type": args.model,
        "task": args.task,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "heads": args.heads,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "seed": args.seed,
        "checkpoint_dir": args.checkpoint_dir,
        "label_names": dataset.label_names,
    }

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    checkpoint_path = Path(args.checkpoint_dir) / "best_model.pt"
    if checkpoint_path.exists():
        load_checkpoint(str(checkpoint_path), model, device=device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, device, criterion
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Compute detailed metrics
    import numpy as np
    from gnn.utils import compute_metrics, get_classification_report

    metrics = compute_metrics(
        np.array(test_labels),
        np.array(test_preds),
        average="weighted",
    )
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

    print(f"\nClassification Report:")
    # Only include labels that appear in test set
    unique_labels = sorted(set(test_labels) | set(test_preds))
    test_label_names = [dataset.label_names[i] for i in unique_labels] if dataset.label_names else None
    print(get_classification_report(
        np.array(test_labels),
        np.array(test_preds),
        label_names=test_label_names,
    ))


if __name__ == "__main__":
    main()
