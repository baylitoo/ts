"""
GNN-based State Encoder for RL Agent

Encodes graph structure and node features into state representations
for the DQN agent. Supports multiple GNN architectures including RMGANets.
"""

import networkx as nx
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv, TransformerConv

from .gnn_modules import RMGANetsEncoder


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for learning node embeddings.

    Aggregates features from neighbors using SAGE convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of SAGE layers
            dropout: Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Build SAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.

    Uses attention mechanism to weight neighbor contributions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Build GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )

        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class TimeEncoding(nn.Module):
    """Sinusoidal time encoding used by temporal GNNs."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError("TimeEncoding dimension must be even.")
        self.dimension = dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dimension, 2).float() / dimension))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: Tensor of shape (num_edges,) containing float timestamps.
        Returns:
            Sinusoidal encodings with shape (num_edges, dimension).
        """
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        sinusoid_inp = timestamps * self.inv_freq
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        return torch.cat([sin, cos], dim=-1)


class TGATEncoder(nn.Module):
    """
    Temporal Graph Attention (TGAT) encoder leveraging time-aware attention.
    """

    supports_temporal = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.dropout = dropout
        self.time_encoder = TimeEncoding(time_dim)

        self.convs = nn.ModuleList()
        self.convs.append(
            TransformerConv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=time_dim,
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    hidden_channels,
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=time_dim,
                )
            )
        self.convs.append(
            TransformerConv(
                hidden_channels,
                out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=time_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return x.new_zeros(x.size(0), self.convs[-1].out_channels)

        if edge_time is None:
            edge_attr = torch.zeros(
                num_edges, self.time_dim, device=x.device, dtype=x.dtype
            )
        else:
            edge_attr = self.time_encoder(edge_time.to(x.device))

        out = x
        for i, conv in enumerate(self.convs):
            out = conv(out, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                out = F.elu(out)
                out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class TGNEncoder(nn.Module):
    """
    Simplified Temporal Graph Network encoder with GRU-style memory updates.
    """

    supports_temporal = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        time_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.hidden_channels = hidden_channels
        self.time_encoder = TimeEncoding(time_dim)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + time_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = x.device
        num_nodes = x.size(0)
        hidden = self.input_proj(x)

        if edge_index.numel() == 0:
            return self.readout(hidden)

        if edge_time is None:
            times = torch.zeros(edge_index.size(1), device=device, dtype=hidden.dtype)
        else:
            times = edge_time.to(device, dtype=hidden.dtype)

        order = torch.argsort(times)
        hidden = hidden.clone()

        for idx in order:
            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()
            time_feat = self.time_encoder(times[idx].unsqueeze(0)).to(device).squeeze(0)

            message_input = torch.cat(
                (hidden[src], hidden[dst], time_feat), dim=-1
            )
            message = self.message_mlp(message_input)

            hidden[src] = self.gru(message, hidden[src])
            hidden[dst] = self.gru(message, hidden[dst])

        return self.readout(hidden)


# ---------------------------------------------------------------------------
# GNN Factory
# ---------------------------------------------------------------------------

def create_gnn(
    gnn_type: str,
    node_feature_dim: int,
    embedding_dim: int,
    device: str = "cpu",
    multi_branch: bool = False,
    use_dqn_enhancement: bool = False,
    num_classes: int = 2,
) -> tuple["nn.Module", bool]:
    """
    Instantiate the requested GNN encoder and move it to *device*.

    Returns:
        (gnn_module, multi_branch_enabled)

    Raises:
        ValueError: for unknown *gnn_type* strings.
    """
    from .gnn_modules.rmganets_encoder_multibranch import RMGANetsMultiBranchEncoder

    multi_branch_enabled = False

    if gnn_type == "sage":
        gnn = GraphSAGEEncoder(
            in_channels=node_feature_dim,
            hidden_channels=64,
            out_channels=embedding_dim,
            num_layers=2,
        )
    elif gnn_type == "gat":
        gnn = GATEncoder(
            in_channels=node_feature_dim,
            hidden_channels=64,
            out_channels=embedding_dim,
            num_layers=2,
            heads=4,
        )
    elif gnn_type == "rmganets":
        if multi_branch:
            gnn = RMGANetsMultiBranchEncoder(
                node_feature_dim=node_feature_dim,
                hidden_dim=160,
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                num_att_heads=8,
                T1=0.7,
                T2=0.3,
                dropout=0.1,
                multi_branch=True,
                use_dqn_enhancement=use_dqn_enhancement,
            )
            multi_branch_enabled = True
        else:
            gnn = RMGANetsEncoder(
                node_feature_dim=node_feature_dim,
                hidden_dim=160,
                embedding_dim=embedding_dim,
                num_att_heads=8,
                T1=0.7,
                T2=0.3,
                dropout=0.1,
                use_dqn_enhancement=use_dqn_enhancement,
            )
    elif gnn_type == "tgat":
        gnn = TGATEncoder(
            in_channels=node_feature_dim,
            hidden_channels=128,
            out_channels=embedding_dim,
            num_layers=3,
            heads=4,
            dropout=0.1,
        )
    elif gnn_type == "tgn":
        gnn = TGNEncoder(
            in_channels=node_feature_dim,
            hidden_channels=128,
            out_channels=embedding_dim,
            time_dim=32,
            dropout=0.1,
        )
    else:
        raise ValueError(
            f"Unknown GNN type: {gnn_type!r}. "
            "Expected one of: 'sage', 'gat', 'rmganets', 'tgat', 'tgn'."
        )

    return gnn.to(device), multi_branch_enabled


class StateEncoder:
    """
    Encodes graph state for RL agent.

    Combines:
    1. GNN node embeddings (current node + subgraph)
    2. Path history features
    3. Temporal features
    4. Statistical features
    """

    def __init__(
        self,
        node_feature_dim: int,
        gnn_type: str = "sage",  # "sage", "gat", or "rmganets"
        embedding_dim: int = 32,
        history_dim: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        multi_branch: bool = False,
        use_dqn_enhancement: bool = False,
        num_classes: int = 2,
        time_attributes: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            node_feature_dim: Dimension of raw node features
            gnn_type: Type of GNN ("sage", "gat", or "rmganets")
            embedding_dim: Dimension of GNN embeddings
            history_dim: Dimension of history encoding
            device: Device for computation
            multi_branch: Enable multi-branch outputs (RMGANets only)
            use_dqn_enhancement: Enable optional DQN enhancement layer
            num_classes: Number of classes for auxiliary heads
        """
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        self.history_dim = history_dim
        self.device = device
        self.gnn_type = gnn_type
        self.multi_branch_enabled = False
        self.num_classes = num_classes
        self.use_dqn_enhancement = use_dqn_enhancement
        self.supports_temporal = False
        default_time_attributes = ("timestamp", "time", "step", "t", "datetime", "ts")
        if time_attributes:
            cleaned = [attr.strip() for attr in time_attributes if attr and attr.strip()]
            self.time_attributes = tuple(dict.fromkeys(cleaned)) or default_time_attributes
        else:
            self.time_attributes = default_time_attributes

        # Initialize GNN via centralised factory
        self.gnn, self.multi_branch_enabled = create_gnn(
            gnn_type=gnn_type,
            node_feature_dim=node_feature_dim,
            embedding_dim=embedding_dim,
            device=device,
            multi_branch=multi_branch,
            use_dqn_enhancement=use_dqn_enhancement,
            num_classes=num_classes,
        )

        self.gnn.eval()  # Start in eval mode
        self.supports_temporal = getattr(self.gnn, "supports_temporal", False)

        # State dimension = GNN embedding + history features
        self.state_dim = embedding_dim + history_dim

    def train(self) -> None:
        """Set GNN to training mode"""
        self.gnn.train()

    def eval(self) -> None:
        """Set GNN to eval mode"""
        self.gnn.eval()

    def forward_state(
        self,
        graph: nx.DiGraph,
        current_node: str,
        visited_nodes: set,
        visited_edges: list,
        node_feature_extractor: callable,
        return_auxiliary: bool = False
    ) -> tuple[torch.Tensor, np.ndarray, Optional[dict]]:
        """Forward pass returning embedding tensor, history features, and optional auxiliary data."""
        subgraph_nodes = self._get_local_subgraph(graph, current_node, k=2)
        subgraph = graph.subgraph(subgraph_nodes)

        pyg_data = self._networkx_to_pyg(subgraph, node_feature_extractor)
        x = pyg_data.x.to(self.device)
        edge_index = pyg_data.edge_index.to(self.device)
        edge_time = None
        if self.supports_temporal and hasattr(pyg_data, "edge_time"):
            edge_attr_tensor = getattr(pyg_data, "edge_time")
            edge_time = edge_attr_tensor.to(self.device)

        node_list = list(subgraph_nodes)
        aux_selected: Optional[dict] = None

        if return_auxiliary and self.multi_branch_enabled:
            node_embeddings, aux_full = self.gnn(x, edge_index, return_auxiliary=True)
            node_idx = node_list.index(current_node)
            aux_selected = {
                "to_branch_logits": aux_full["to_branch"][node_idx],
                "hy_branch_logits": aux_full["hy_branch"][node_idx],
                "subgraph_stats": aux_full.get("subgraph_stats"),
            }
        else:
            # For non-multi-branch GNNs or when not requesting auxiliary outputs
            if self.multi_branch_enabled:
                node_embeddings = self.gnn(x, edge_index, return_auxiliary=False)
            elif self.supports_temporal:
                node_embeddings = self.gnn(x, edge_index, edge_time=edge_time)
            else:
                node_embeddings = self.gnn(x, edge_index)

        node_idx = node_list.index(current_node)
        embedding = node_embeddings[node_idx]
        history_features = self._encode_history(graph, visited_nodes, visited_edges)

        return embedding, history_features, aux_selected

    def encode_state(
        self,
        graph: nx.DiGraph,
        current_node: str,
        visited_nodes: set,
        visited_edges: list,
        node_feature_extractor: callable
    ) -> np.ndarray:
        """
        Encode current state as feature vector

        Args:
            graph: Full transaction graph
            current_node: Current node ID
            visited_nodes: Set of visited node IDs
            visited_edges: List of visited edges
            node_feature_extractor: Function to extract features from node

        Returns:
            State vector (state_dim,)
        """
        with torch.no_grad():
            embedding, history_features, _ = self.forward_state(
                graph,
                current_node,
                visited_nodes,
                visited_edges,
                node_feature_extractor,
                return_auxiliary=False
            )
        current_embedding = embedding.cpu().numpy()
        state = np.concatenate([current_embedding, history_features])

        return state

    def _get_local_subgraph(
        self,
        graph: nx.DiGraph,
        center_node: str,
        k: int = 2
    ) -> set:
        """Get k-hop neighborhood around node"""
        neighbors = {center_node}

        for _ in range(k):
            new_neighbors = set()
            for node in neighbors:
                if node in graph:
                    new_neighbors.update(graph.predecessors(node))
                    new_neighbors.update(graph.successors(node))
            neighbors.update(new_neighbors)

        return neighbors

    def _networkx_to_pyg(
        self,
        graph: nx.DiGraph,
        node_feature_extractor: callable
    ) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data"""
        # Node mapping
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Extract node features
        node_features = []
        for node in node_list:
            node_data = graph.nodes[node]
            if hasattr(node_feature_extractor, "extract"):
                features = node_feature_extractor.extract(node_data)
            else:
                features = node_feature_extractor(node_data)
            node_features.append(features)

        x = torch.FloatTensor(np.array(node_features))
        if x.numel() > 0 and x.size(1) != self.node_feature_dim:
            raise ValueError(
                f"Node feature dimension mismatch: expected {self.node_feature_dim}, got {x.size(1)}. "
                "Ensure NodeFeatureExtractor.node_feature_dim matches GNNConfig.node_feature_dim."
            )

        # Extract edges and timestamps
        edge_list = []
        edge_times = []
        track_time = self.supports_temporal
        for u, v, data in graph.edges(data=True):
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            if track_time:
                time_val = None
                for key in self.time_attributes:
                    if key in data:
                        time_val = data[key]
                        break
                if time_val is None:
                    edge_times.append(0.0)
                else:
                    try:
                        edge_times.append(float(time_val))
                    except (TypeError, ValueError):
                        edge_times.append(0.0)

        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_time_tensor = torch.tensor(edge_times, dtype=torch.float32) if track_time else None
        else:
            edge_index = torch.LongTensor([[], []])
            edge_time_tensor = torch.zeros(0, dtype=torch.float32) if track_time else None

        data = Data(x=x, edge_index=edge_index)
        if self.supports_temporal and edge_time_tensor is not None:
            data.edge_time = edge_time_tensor
        return data

    def _encode_history(
        self,
        graph: nx.DiGraph,
        visited_nodes: set,
        visited_edges: list
    ) -> np.ndarray:
        """
        Encode path history as features

        Returns:
            History vector (history_dim,)
        """
        features = []

        # 1. Basic statistics
        num_visited = len(visited_nodes)
        num_edges = len(visited_edges)
        features.extend([num_visited / 100.0, num_edges / 100.0])  # Normalized

        # 2. Fraud signals
        num_fraud_edges = sum(
            1 for u, v in visited_edges
            if graph.get_edge_data(u, v, {}).get('is_fraud', False)
            or graph.get_edge_data(u, v, {}).get('is_money_laundering', False)
            or graph.get_edge_data(u, v, {}).get('isFraud', False)
        )
        fraud_rate = num_fraud_edges / max(num_edges, 1)
        features.append(fraud_rate)

        # 3. Amount statistics
        if visited_edges:
            amounts = [
                graph.get_edge_data(u, v, {}).get('amount', 0.0)
                for u, v in visited_edges
            ]
            features.extend([
                np.log1p(np.mean(amounts)),
                np.log1p(np.std(amounts)),
                np.log1p(np.max(amounts))
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 4. Temporal pattern (if available)
        if visited_edges:
            hours = [
                graph.get_edge_data(u, v, {}).get('hour', 12)
                for u, v in visited_edges
            ]
            avg_hour = np.mean(hours)
            features.extend([
                np.sin(2 * np.pi * avg_hour / 24),
                np.cos(2 * np.pi * avg_hour / 24)
            ])
        else:
            features.extend([0.0, 0.0])

        # 5. Risk scores (if available)
        if visited_nodes:
            risk_scores = [
                graph.nodes[node].get('risk_score', 0.0)
                for node in visited_nodes
                if node in graph
            ]
            if risk_scores:
                features.extend([
                    np.mean(risk_scores),
                    np.max(risk_scores)
                ])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])

        # Pad or truncate to history_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.history_dim:
            features = np.pad(features, (0, self.history_dim - len(features)))
        elif len(features) > self.history_dim:
            features = features[:self.history_dim]

        return features

    def get_state_dim(self) -> int:
        """Get total state dimension"""
        return self.state_dim
