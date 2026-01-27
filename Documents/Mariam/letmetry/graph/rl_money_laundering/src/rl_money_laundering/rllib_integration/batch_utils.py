"""
Batch processing utilities for graph observations.

Provides efficient batching of variable-size graphs using PyTorch Geometric's Batch class.
This is ~10x faster than sequential processing.
"""

from typing import Dict, List
import torch
from torch_geometric.data import Data, Batch


def batch_graph_observations(graph_obs_batch: Dict[str, torch.Tensor]) -> Batch:
    """
    Convert batched GraphSpace observations to PyTorch Geometric Batch.

    This enables efficient parallel processing of multiple graphs through GNN layers.

    Args:
        graph_obs_batch: Dict with keys:
            - node_features: (batch_size, max_nodes, node_feature_dim)
            - edge_index: (batch_size, 2, max_edges)
            - edge_time: (batch_size, max_edges)
            - num_nodes: (batch_size,)
            - num_edges: (batch_size,)
            - context_features: (batch_size, context_feature_dim)

    Returns:
        PyG Batch object with:
            - x: (total_nodes, node_feature_dim) - concatenated node features
            - edge_index: (2, total_edges) - concatenated edge indices (offset by batch)
            - batch: (total_nodes,) - batch assignment for each node
            - ptr: (batch_size + 1,) - pointer to start of each graph
            - context: (batch_size, context_feature_dim) - context features per graph

    Example:
        >>> # Input: 3 graphs with varying sizes
        >>> graph_obs = {
        ...     'node_features': torch.randn(3, 10, 8),
        ...     'num_nodes': torch.tensor([5, 3, 7]),
        ...     'num_edges': torch.tensor([8, 4, 10]),
        ...     ...
        ... }
        >>> pyg_batch = batch_graph_observations(graph_obs)
        >>> pyg_batch.x.shape  # (15, 8) - 5+3+7 nodes total
        >>> pyg_batch.batch  # [0,0,0,0,0, 1,1,1, 2,2,2,2,2,2,2]
    """
    batch_size = graph_obs_batch["node_features"].shape[0]
    device = graph_obs_batch["node_features"].device

    # Extract components
    node_features_padded = graph_obs_batch["node_features"]  # (B, max_nodes, feat_dim)
    edge_index_padded = graph_obs_batch["edge_index"]  # (B, 2, max_edges)
    edge_time_padded = graph_obs_batch.get("edge_time")  # (B, max_edges) or None
    num_nodes = graph_obs_batch["num_nodes"]  # (B,) or (B, 1)
    num_edges = graph_obs_batch["num_edges"]  # (B,) or (B, 1)
    context_features = graph_obs_batch["context_features"]  # (B, context_dim)

    # Handle both (B,) and (B, 1) shapes for num_nodes/num_edges
    if num_nodes.ndim == 2:
        num_nodes = num_nodes.squeeze(1)  # (B, 1) -> (B,)
    if num_edges.ndim == 2:
        num_edges = num_edges.squeeze(1)  # (B, 1) -> (B,)

    # Create individual Data objects for each graph
    data_list: List[Data] = []
    for i in range(batch_size):
        n_nodes = int(num_nodes[i].item())
        n_edges = int(num_edges[i].item())

        # Extract active nodes and edges (remove padding)
        x = node_features_padded[i, :n_nodes]  # (n_nodes, feat_dim)
        edge_idx = edge_index_padded[i, :, :n_edges]  # (2, n_edges)
        edge_time = None
        if edge_time_padded is not None:
            edge_time = edge_time_padded[i, :n_edges].to(dtype=torch.float32)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_idx,
            num_nodes=n_nodes,
        )
        if edge_time is not None:
            data.edge_time = edge_time
        data_list.append(data)

    # Batch graphs together (PyG handles index offsetting automatically)
    pyg_batch = Batch.from_data_list(data_list)

    # Add context features (not part of standard PyG, but useful for our case)
    pyg_batch.context = context_features

    return pyg_batch


def unbatch_node_embeddings(
    node_embeddings: torch.Tensor,
    batch_assignment: torch.Tensor,
    batch_size: int
) -> List[torch.Tensor]:
    """
    Split batched node embeddings back into per-graph embeddings.

    Args:
        node_embeddings: (total_nodes, embedding_dim) - concatenated embeddings
        batch_assignment: (total_nodes,) - batch index for each node
        batch_size: Number of graphs in batch

    Returns:
        List of (num_nodes_i, embedding_dim) tensors, one per graph

    Example:
        >>> embeddings = torch.randn(15, 32)  # 15 total nodes
        >>> batch = torch.tensor([0,0,0,0,0, 1,1,1, 2,2,2,2,2,2,2])
        >>> per_graph = unbatch_node_embeddings(embeddings, batch, 3)
        >>> [e.shape for e in per_graph]
        [torch.Size([5, 32]), torch.Size([3, 32]), torch.Size([7, 32])]
    """
    embeddings_list = []

    for i in range(batch_size):
        # Extract embeddings for graph i
        mask = batch_assignment == i
        graph_embeddings = node_embeddings[mask]
        embeddings_list.append(graph_embeddings)

    return embeddings_list


def extract_first_node_embeddings(
    node_embeddings: torch.Tensor,
    batch_assignment: torch.Tensor,
    batch_size: int
) -> torch.Tensor:
    """
    Extract first node embedding from each graph in batch.

    In our case, the first node is always the current node of interest.

    Args:
        node_embeddings: (total_nodes, embedding_dim)
        batch_assignment: (total_nodes,)
        batch_size: Number of graphs

    Returns:
        (batch_size, embedding_dim) - first node embedding per graph

    Example:
        >>> embeddings = torch.randn(15, 32)
        >>> batch = torch.tensor([0,0,0,0,0, 1,1,1, 2,2,2,2,2,2,2])
        >>> first_nodes = extract_first_node_embeddings(embeddings, batch, 3)
        >>> first_nodes.shape
        torch.Size([3, 32])
    """
    device = node_embeddings.device
    embedding_dim = node_embeddings.shape[1]
    first_embeddings = torch.zeros(batch_size, embedding_dim, device=device)

    for i in range(batch_size):
        # Find first node of graph i
        mask = batch_assignment == i
        indices = torch.where(mask)[0]
        if len(indices) > 0:
            first_idx = indices[0]
            first_embeddings[i] = node_embeddings[first_idx]

    return first_embeddings


def batch_temporal_edges(
    edge_time_list: List[torch.Tensor],
    num_edges_per_graph: List[int]
) -> torch.Tensor:
    """
    Batch temporal edge attributes (timestamps).

    Args:
        edge_time_list: List of (num_edges_i,) tensors
        num_edges_per_graph: Number of edges in each graph

    Returns:
        (total_edges,) concatenated edge times
    """
    # Filter out None and empty tensors
    valid_times = [
        t for t, n in zip(edge_time_list, num_edges_per_graph)
        if t is not None and n > 0
    ]

    if not valid_times:
        return None

    return torch.cat(valid_times, dim=0)


class GraphBatchCollator:
    """
    Collator for efficiently batching graph observations.

    Usage:
        collator = GraphBatchCollator()
        pyg_batch = collator(graph_obs_batch)
        node_embeddings = gnn(pyg_batch.x, pyg_batch.edge_index)
        first_node_embeds = collator.extract_first_nodes(node_embeddings, pyg_batch)
    """

    def __call__(self, graph_obs_batch: Dict[str, torch.Tensor]) -> Batch:
        """Convert graph observations to PyG Batch."""
        return batch_graph_observations(graph_obs_batch)

    def extract_first_nodes(
        self,
        node_embeddings: torch.Tensor,
        pyg_batch: Batch
    ) -> torch.Tensor:
        """Extract first node embedding from each graph."""
        # Get batch size from num_graphs or context if available
        if hasattr(pyg_batch, 'context'):
            batch_size = pyg_batch.context.shape[0]
        elif hasattr(pyg_batch, 'num_graphs'):
            batch_size = pyg_batch.num_graphs
        else:
            batch_size = int(pyg_batch.batch.max().item()) + 1

        return extract_first_node_embeddings(
            node_embeddings,
            pyg_batch.batch,
            batch_size
        )

    def extract_all_nodes(
        self,
        node_embeddings: torch.Tensor,
        pyg_batch: Batch
    ) -> List[torch.Tensor]:
        """Split node embeddings by graph."""
        # Get batch size from num_graphs or context if available
        if hasattr(pyg_batch, 'context'):
            batch_size = pyg_batch.context.shape[0]
        elif hasattr(pyg_batch, 'num_graphs'):
            batch_size = pyg_batch.num_graphs
        else:
            batch_size = int(pyg_batch.batch.max().item()) + 1

        return unbatch_node_embeddings(
            node_embeddings,
            pyg_batch.batch,
            batch_size
        )
