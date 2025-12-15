"""
Utility functions for PackHero GNN.

Includes:
- Device management (GPU/CPU)
- Random seed setting for reproducibility
- Feature normalization
- Metrics computation
- Graph visualization
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Get the best available device (GPU if available, else CPU).

    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def normalize_features(
    x: torch.Tensor,
    method: str = "standard",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize node features.

    Args:
        x: Node feature tensor of shape (num_nodes, num_features)
        method: Normalization method ('standard', 'minmax', 'l2')
        eps: Small constant for numerical stability

    Returns:
        Normalized feature tensor
    """
    if method == "standard":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + eps)
    elif method == "minmax":
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        return (x - x_min) / (x_max - x_min + eps)
    elif method == "l2":
        norm = x.norm(dim=1, keepdim=True)
        return x / (norm + eps)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label names
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None
) -> str:
    """
    Generate a detailed classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: Optional list of label names

    Returns:
        Formatted classification report string
    """
    return classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: Optional list of label names
        save_path: Optional path to save the figure
        figsize: Figure size
        cmap: Colormap name

    Returns:
        matplotlib Figure object
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    if label_names is not None:
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=label_names,
            yticklabels=label_names,
        )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    axes[0].plot(history.get("train_loss", []), label="Train Loss", marker="o")
    axes[0].plot(history.get("val_loss", []), label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history.get("train_acc", []), label="Train Acc", marker="o")
    axes[1].plot(history.get("val_acc", []), label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")

    return fig


def visualize_graph(
    data: Any,
    node_labels: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    node_size: int = 500,
    font_size: int = 8
) -> Optional[plt.Figure]:
    """
    Visualize a graph using networkx.

    Args:
        data: PyTorch Geometric Data object
        node_labels: Optional dictionary mapping node indices to labels
        save_path: Optional path to save the figure
        figsize: Figure size
        node_size: Size of nodes in visualization
        font_size: Font size for labels

    Returns:
        matplotlib Figure object or None if networkx not available
    """
    try:
        import networkx as nx
        from torch_geometric.utils import to_networkx
    except ImportError:
        print("networkx required for graph visualization. Install with: pip install networkx")
        return None

    # Convert to networkx graph
    G = to_networkx(data, to_undirected=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Use spring layout for visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw the graph
    nx.draw(
        G, pos,
        ax=ax,
        node_size=node_size,
        node_color="lightblue",
        edge_color="gray",
        alpha=0.7,
        with_labels=True,
        font_size=font_size
    )

    ax.set_title(f"Call Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graph visualization saved to: {save_path}")

    return fig


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to map checkpoint to

    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from: {path} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
