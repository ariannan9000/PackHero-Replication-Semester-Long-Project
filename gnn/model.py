"""
Graph Neural Network models for packer classification.

Implements GCN and GAT-based architectures for classifying PE executables
based on their call graph structure.

Architecture:
    Input -> GNN Layers -> Global Pooling -> MLP Classifier -> Output

Based on the PackHero methodology for packer identification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch_geometric.data import Data, Batch


@dataclass
class PackerGNNConfig:
    """
    Configuration for PackerGNN model.

    Attributes:
        in_channels: Number of input node features (default: 5 for PackHero features)
        hidden_channels: Hidden layer dimensions
        num_classes: Number of output classes
        num_layers: Number of GNN layers
        conv_type: Type of graph convolution ('gcn', 'gat', 'sage', 'gin')
        pooling: Global pooling method ('mean', 'max', 'add')
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
        heads: Number of attention heads (for GAT only)
        mlp_hidden: MLP classifier hidden dimensions
    """
    in_channels: int = 5
    hidden_channels: int = 64
    num_classes: int = 4
    num_layers: int = 3
    conv_type: Literal["gcn", "gat", "sage", "gin"] = "gcn"
    pooling: Literal["mean", "max", "add"] = "mean"
    dropout: float = 0.5
    batch_norm: bool = True
    heads: int = 4
    mlp_hidden: List[int] = field(default_factory=lambda: [128, 64])


class GNNLayer(nn.Module):
    """
    A single GNN layer with optional batch normalization and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = "gcn",
        batch_norm: bool = True,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        super().__init__()

        # Create convolution layer based on type
        if conv_type == "gcn":
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == "gat":
            # GAT outputs heads * out_channels, so we adjust
            self.conv = GATConv(in_channels, out_channels // heads, heads=heads, concat=True)
            out_channels = (out_channels // heads) * heads  # Adjust for concat
        elif conv_type == "sage":
            self.conv = SAGEConv(in_channels, out_channels)
        elif conv_type == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.conv = GINConv(mlp)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN layer.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        x = self.conv(x, edge_index)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class PackerGNN(nn.Module):
    """
    Graph Neural Network for packer classification.

    Architecture:
        1. Input projection layer
        2. Stack of GNN layers (GCN/GAT/SAGE/GIN)
        3. Global pooling (mean/max/add)
        4. MLP classifier
    """

    def __init__(self, config: Optional[PackerGNNConfig] = None, **kwargs):
        """
        Initialize the PackerGNN model.

        Args:
            config: Model configuration
            **kwargs: Override config parameters
        """
        super().__init__()

        # Create config
        if config is None:
            config = PackerGNNConfig(**kwargs)
        else:
            # Allow overriding config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.in_channels, config.hidden_channels)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(config.num_layers):
            in_ch = config.hidden_channels
            out_ch = config.hidden_channels
            self.gnn_layers.append(
                GNNLayer(
                    in_ch, out_ch,
                    conv_type=config.conv_type,
                    batch_norm=config.batch_norm,
                    dropout=config.dropout,
                    heads=config.heads,
                )
            )

        # Global pooling
        if config.pooling == "mean":
            self.pool = global_mean_pool
        elif config.pooling == "max":
            self.pool = global_max_pool
        elif config.pooling == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling: {config.pooling}")

        # MLP classifier
        mlp_layers = []
        prev_dim = config.hidden_channels

        for hidden_dim in config.mlp_hidden:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = hidden_dim

        mlp_layers.append(nn.Linear(prev_dim, config.num_classes))
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes] for batched graphs

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Handle empty graphs
        if x.size(0) == 0:
            batch_size = 1 if batch is None else batch.max().item() + 1
            return torch.zeros(batch_size, self.config.num_classes, device=x.device)

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.pool(x, batch)

        # Classification
        out = self.classifier(x)

        return out

    def forward_batch(self, data: Data) -> torch.Tensor:
        """
        Forward pass for a PyTorch Geometric Data/Batch object.

        Args:
            data: PyTorch Geometric Data or Batch object

        Returns:
            Class logits [batch_size, num_classes]
        """
        return self.forward(
            data.x,
            data.edge_index,
            data.batch if hasattr(data, 'batch') else None
        )

    def predict(self, data: Data) -> torch.Tensor:
        """
        Get predictions for a graph.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Predicted class indices [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward_batch(data)
            return logits.argmax(dim=-1)

    def predict_proba(self, data: Data) -> torch.Tensor:
        """
        Get prediction probabilities for a graph.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Class probabilities [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward_batch(data)
            return F.softmax(logits, dim=-1)


class PackerGMN(nn.Module):
    """
    Graph Matching Network for packer identification.

    Based on the original PackHero paper's approach using graph similarity.
    Computes embeddings for query graphs and compares with reference graphs.

    This is a simplified version - the full GMN would include:
    - Cross-graph attention
    - Graph-level matching vectors
    - Siamese network structure
    """

    def __init__(self, config: Optional[PackerGNNConfig] = None, **kwargs):
        """
        Initialize the Graph Matching Network.

        Args:
            config: Model configuration
            **kwargs: Override config parameters
        """
        super().__init__()

        if config is None:
            config = PackerGNNConfig(**kwargs)

        self.config = config

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(config.in_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # GNN for graph embedding
        self.gnn_layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.gnn_layers.append(
                GNNLayer(
                    config.hidden_channels,
                    config.hidden_channels,
                    conv_type=config.conv_type,
                    batch_norm=config.batch_norm,
                    dropout=config.dropout,
                    heads=config.heads,
                )
            )

        # Graph-level pooling
        self.pool = global_mean_pool

        # Final projection to embedding space
        self.graph_proj = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, config.hidden_channels),
        )

        # Classification head
        self.classifier = nn.Linear(config.hidden_channels, config.num_classes)

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute graph-level embedding.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector

        Returns:
            Graph embedding [batch_size, hidden_channels]
        """
        # Encode nodes
        x = self.node_encoder(x)

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)

        # Pool to graph level
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_emb = self.pool(x, batch)

        # Project
        graph_emb = self.graph_proj(graph_emb)

        return graph_emb

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment vector

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Handle empty graphs
        if x.size(0) == 0:
            batch_size = 1 if batch is None else batch.max().item() + 1
            return torch.zeros(batch_size, self.config.num_classes, device=x.device)

        graph_emb = self.get_graph_embedding(x, edge_index, batch)
        return self.classifier(graph_emb)

    def forward_batch(self, data: Data) -> torch.Tensor:
        """Forward pass for PyTorch Geometric Data/Batch."""
        return self.forward(
            data.x,
            data.edge_index,
            data.batch if hasattr(data, 'batch') else None
        )


def create_model(
    model_type: str = "gcn",
    num_classes: int = 4,
    in_channels: int = 5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a packer classification model.

    Args:
        model_type: Model architecture ('gcn', 'gat', 'sage', 'gin', 'gmn')
        num_classes: Number of output classes
        in_channels: Number of input features
        **kwargs: Additional model configuration

    Returns:
        PyTorch model
    """
    if model_type == "gmn":
        config = PackerGNNConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
        return PackerGMN(config)
    else:
        config = PackerGNNConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_type=model_type,
            **kwargs
        )
        return PackerGNN(config)


if __name__ == "__main__":
    # Test the model
    from dataset import create_sample_graph

    print("Testing PackerGNN model...")

    # Create sample data
    data = create_sample_graph(num_nodes=20, num_features=5, num_edges=30)
    print(f"Sample graph: {data.num_nodes} nodes, {data.num_edges} edges")

    # Test different model types
    for model_type in ["gcn", "gat", "sage", "gin", "gmn"]:
        print(f"\nTesting {model_type.upper()} model:")

        model = create_model(
            model_type=model_type,
            num_classes=4,
            in_channels=5,
            hidden_channels=32,
            num_layers=2,
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward_batch'):
                out = model.forward_batch(data)
            else:
                out = model(data.x, data.edge_index)

        print(f"  Output shape: {out.shape}")
        print(f"  Prediction: {out.argmax(dim=-1).item()}")

    print("\nAll models tested successfully!")
