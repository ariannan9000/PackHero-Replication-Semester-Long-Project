"""
Dataset loader for PackHero GNN.

Handles loading PyTorch Geometric graph files (.pt) and creating
train/validation/test splits for packer classification.

Expected graph format:
- PyTorch Geometric Data or HeteroData objects
- Node features: [size, blocks, instructions, outgoing_calls, is_entry]
- Edge index representing call graph edges
"""

import os
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.loader import DataLoader


# Label mappings
PACKER_LABELS = {
    "unpacked": 0,
    "upx_best": 1,
    "upx_standard": 2,
    "upx_fast": 3,
}

BINARY_LABELS = {
    "unpacked": 0,
    "packed": 1,
}

# Node feature names (in order)
NODE_FEATURES = ["size", "blocks", "instructions", "outgoing_calls", "is_entry"]


class PackerDataset(Dataset):
    """
    PyTorch Geometric Dataset for packer classification.

    Loads pre-extracted graph files (.pt) and their corresponding labels.
    Supports both multi-class (packer family) and binary (packed/unpacked) classification.
    """

    def __init__(
        self,
        root: str,
        graph_dir: Optional[str] = None,
        csv_path: Optional[str] = None,
        task: str = "multiclass",
        transform=None,
        pre_transform=None,
        normalize_features: bool = True,
    ):
        """
        Args:
            root: Root directory of the dataset
            graph_dir: Directory containing .pt graph files (default: root/graphs)
            csv_path: Path to CSV file with labels (default: root/labels/packhero_dataset.csv)
            task: Classification task - 'multiclass' or 'binary'
            transform: PyTorch Geometric transform to apply
            pre_transform: PyTorch Geometric pre-transform to apply
            normalize_features: Whether to normalize node features
        """
        self.root = Path(root)
        self.graph_dir = Path(graph_dir) if graph_dir else self.root / "graphs"
        self.csv_path = Path(csv_path) if csv_path else self.root / "labels" / "packhero_dataset.csv"
        self.task = task
        self.normalize_features = normalize_features

        # Load labels
        self.labels_df = self._load_labels()
        self.samples = self._find_graph_files()

        # Get label mapping
        self.label_map = BINARY_LABELS if task == "binary" else PACKER_LABELS
        self._num_classes = len(set(self.label_map.values()))
        self.label_names = list(self.label_map.keys())

        super().__init__(str(self.root), transform, pre_transform)

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self._num_classes

    def _load_labels(self) -> pd.DataFrame:
        """Load and parse the labels CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Labels CSV not found: {self.csv_path}\n"
                "Please ensure the dataset CSV exists."
            )

        df = pd.read_csv(self.csv_path)

        # Ensure required columns exist
        required_cols = ["filename", "packer", "label"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        return df

    def _find_graph_files(self) -> List[Dict]:
        """Find all graph files and match with labels."""
        samples = []

        if not self.graph_dir.exists():
            print(f"Warning: Graph directory not found: {self.graph_dir}")
            print("Creating directory structure. Please add .pt graph files.")
            self.graph_dir.mkdir(parents=True, exist_ok=True)
            return samples

        # Find all .pt files
        pt_files = list(self.graph_dir.glob("*.pt"))

        if not pt_files:
            print(f"Warning: No .pt files found in {self.graph_dir}")
            print("Please run graph extraction to generate .pt files.")
            return samples

        # Match with labels
        for pt_file in pt_files:
            filename = pt_file.stem  # filename without extension

            # Try to find matching label
            matching_rows = self.labels_df[
                self.labels_df["filename"].str.contains(filename, regex=False)
            ]

            if len(matching_rows) == 0:
                # Try matching by hash (files might be named by hash)
                matching_rows = self.labels_df[
                    self.labels_df["filename"].str.contains(filename[:16], regex=False)
                ]

            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                packer = row["packer"]
                binary_label = row["label"]

                samples.append({
                    "path": pt_file,
                    "filename": filename,
                    "packer": packer,
                    "binary_label": binary_label,
                })
            else:
                print(f"Warning: No label found for {filename}")

        print(f"Found {len(samples)} graph files with labels")
        return samples

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return [s["path"].name for s in self.samples]

    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return self.raw_file_names

    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def get(self, idx: int) -> Data:
        """
        Get a single graph sample.

        Args:
            idx: Sample index

        Returns:
            PyTorch Geometric Data object with graph and label
        """
        sample = self.samples[idx]
        path = sample["path"]

        # Load the graph (weights_only=False needed for PyG Data objects)
        data = torch.load(path, weights_only=False)

        # Handle HeteroData format
        if isinstance(data, HeteroData):
            data = self._convert_heterodata(data)

        # Add label
        if self.task == "binary":
            label = BINARY_LABELS.get(sample["binary_label"], 0)
        else:
            label = PACKER_LABELS.get(sample["packer"], 0)

        data.y = torch.tensor([label], dtype=torch.long)

        # Normalize features if requested
        if self.normalize_features and data.x is not None:
            data.x = self._normalize(data.x)

        # Add metadata
        data.filename = sample["filename"]
        data.packer = sample["packer"]

        return data

    def _convert_heterodata(self, hetero_data: HeteroData) -> Data:
        """
        Convert HeteroData to regular Data object.

        Assumes the graph has a single node type and edge type.
        """
        # Try to get node features
        x = None
        for node_type in hetero_data.node_types:
            if hasattr(hetero_data[node_type], 'x'):
                x = hetero_data[node_type].x
                break

        # Try to get edge index
        edge_index = None
        for edge_type in hetero_data.edge_types:
            if hasattr(hetero_data[edge_type], 'edge_index'):
                edge_index = hetero_data[edge_type].edge_index
                break

        # Create regular Data object
        data = Data(x=x, edge_index=edge_index)

        # Copy any additional attributes
        for key in hetero_data.keys:
            if key not in ['x', 'edge_index'] and not hasattr(data, key):
                setattr(data, key, hetero_data[key])

        return data

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize node features using z-score normalization."""
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + 1e-8)

    def get_label_distribution(self) -> Dict[str, int]:
        """Get the distribution of labels in the dataset."""
        distribution = {}
        for sample in self.samples:
            if self.task == "binary":
                label = sample["binary_label"]
            else:
                label = sample["packer"]
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


def get_dataloaders(
    root: str,
    graph_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    task: str = "multiclass",
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0,
    normalize_features: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, PackerDataset]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        root: Root directory of the dataset
        graph_dir: Directory containing .pt graph files
        csv_path: Path to CSV file with labels
        task: Classification task - 'multiclass' or 'binary'
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility
        num_workers: Number of workers for DataLoaders
        normalize_features: Whether to normalize node features

    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    # Create dataset
    dataset = PackerDataset(
        root=root,
        graph_dir=graph_dir,
        csv_path=csv_path,
        task=task,
        normalize_features=normalize_features,
    )

    if len(dataset) == 0:
        raise ValueError(
            "Dataset is empty. Please ensure:\n"
            "1. Graph files (.pt) exist in the graphs directory\n"
            "2. Labels CSV contains matching filenames\n"
            f"Graph dir: {dataset.graph_dir}\n"
            f"CSV path: {dataset.csv_path}"
        )

    # Get indices and labels for stratified split
    indices = list(range(len(dataset)))
    labels = []
    for sample in dataset.samples:
        if task == "binary":
            labels.append(BINARY_LABELS.get(sample["binary_label"], 0))
        else:
            labels.append(PACKER_LABELS.get(sample["packer"], 0))

    # Split into train and temp (val + test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels if len(set(labels)) > 1 else None
    )

    # Split temp into val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, dataset


def create_sample_graph(
    num_nodes: int = 10,
    num_features: int = 5,
    num_edges: int = 15,
    seed: int = 42
) -> Data:
    """
    Create a sample graph for testing.

    This is useful for testing the GNN model before actual graph extraction.

    Args:
        num_nodes: Number of nodes in the graph
        num_features: Number of node features
        num_edges: Number of edges
        seed: Random seed

    Returns:
        PyTorch Geometric Data object
    """
    torch.manual_seed(seed)

    # Random node features
    x = torch.randn(num_nodes, num_features)

    # Random edges (ensure valid indices)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Random label
    y = torch.tensor([0], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


if __name__ == "__main__":
    # Test dataset loading
    import argparse

    parser = argparse.ArgumentParser(description="Test PackerDataset")
    parser.add_argument("--root", type=str, default="packhero-dataset", help="Dataset root directory")
    parser.add_argument("--task", type=str, default="multiclass", choices=["multiclass", "binary"])
    args = parser.parse_args()

    print("Testing PackerDataset...")
    print(f"Root: {args.root}")
    print(f"Task: {args.task}")

    try:
        dataset = PackerDataset(root=args.root, task=args.task)
        print(f"\nDataset size: {len(dataset)}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Label names: {dataset.label_names}")
        print(f"\nLabel distribution:")
        for label, count in dataset.get_label_distribution().items():
            print(f"  {label}: {count}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample graph:")
            print(f"  Nodes: {sample.num_nodes}")
            print(f"  Edges: {sample.num_edges}")
            print(f"  Features: {sample.x.shape if sample.x is not None else 'None'}")
            print(f"  Label: {sample.y.item()}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this dataset, please:")
        print("1. Run graph extraction on PE files to create .pt files")
        print("2. Place .pt files in the 'graphs' subdirectory")
        print("3. Ensure filenames match entries in packhero_dataset.csv")
