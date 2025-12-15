"""
PackHero GNN - Graph Neural Network for Packer Identification

This module implements a GNN-based classifier for identifying packers
in PE executables based on their call graph structure.

Components:
- model.py: GNN architecture (GCN/GAT-based classifier)
- dataset.py: PyTorch Geometric dataset loader
- train.py: Training loop with early stopping
- evaluate.py: Model evaluation and metrics
- utils.py: Helper functions
"""

from .model import PackerGNN, PackerGNNConfig
from .dataset import PackerDataset, get_dataloaders
from .utils import set_seed, get_device

__version__ = "0.1.0"
__all__ = [
    "PackerGNN",
    "PackerGNNConfig",
    "PackerDataset",
    "get_dataloaders",
    "set_seed",
    "get_device",
]
