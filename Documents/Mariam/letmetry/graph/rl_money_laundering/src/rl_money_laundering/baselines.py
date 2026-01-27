"""
Baseline Methods for AML Detection

Implements traditional and ML-based baselines for comparison:
1. PageRank-based anomaly detection
2. Random Forest on hand-crafted features
3. Static GNN (GraphSAGE/GAT) for node classification
"""

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv


class PageRankBaseline:
    """
    PageRank-based anomaly detection.

    Anomaly score = PageRank centrality + out-degree / in-degree ratio
    """

    def __init__(self, alpha: float = 0.85):
        """
        Args:
            alpha: Damping parameter for PageRank
        """
        self.alpha = alpha
        self.scores = {}

    def fit(self, graph: nx.DiGraph) -> None:
        """
        Compute PageRank scores for all nodes

        Args:
            graph: Transaction graph
        """
        print("Computing PageRank scores...")

        # Compute PageRank
        pagerank = nx.pagerank(graph, alpha=self.alpha)

        # Compute degree-based anomaly scores
        for node in graph.nodes():
            pr_score = pagerank.get(node, 0.0)
            out_degree = graph.out_degree(node)
            in_degree = graph.in_degree(node)

            # Anomaly score: high PR + unusual degree ratio
            degree_ratio = out_degree / max(in_degree, 1)

            # Combine scores
            anomaly_score = pr_score * (1 + np.log1p(degree_ratio))

            self.scores[node] = anomaly_score

    def predict(self, nodes: list, threshold: float = 0.5) -> np.ndarray:
        """
        Predict fraud labels based on threshold

        Args:
            nodes: List of node IDs
            threshold: Anomaly score threshold (percentile)

        Returns:
            Binary predictions (1 = fraud)
        """
        scores = np.array([self.scores.get(node, 0.0) for node in nodes])

        # Use percentile-based threshold
        threshold_value = np.percentile(scores, threshold * 100)

        predictions = (scores >= threshold_value).astype(int)

        return predictions

    def get_scores(self, nodes: list) -> np.ndarray:
        """Get anomaly scores for nodes"""
        return np.array([self.scores.get(node, 0.0) for node in nodes])


class RandomForestBaseline:
    """
    Random Forest on hand-crafted graph features.

    Features:
    - Node degree statistics
    - Transaction amount statistics
    - Temporal patterns
    - Network centrality measures
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Args:
            n_estimators: Number of trees
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced',  # Handle imbalance
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def extract_features(self, graph: nx.DiGraph, nodes: list) -> np.ndarray:
        """
        Extract hand-crafted features for nodes

        Args:
            graph: Transaction graph
            nodes: List of node IDs

        Returns:
            Feature matrix (n_nodes, n_features)
        """
        features = []

        for node in nodes:
            if node not in graph:
                # Node not in graph, use zeros
                features.append(np.zeros(20))
                continue

            node_data = graph.nodes[node]
            node_features = []

            # 1. Degree features
            out_degree = graph.out_degree(node)
            in_degree = graph.in_degree(node)
            total_degree = out_degree + in_degree
            degree_ratio = out_degree / max(in_degree, 1)

            node_features.extend([
                np.log1p(out_degree),
                np.log1p(in_degree),
                np.log1p(total_degree),
                degree_ratio
            ])

            # 2. Transaction amount features
            out_edges = graph.out_edges(node, data=True)
            in_edges = graph.in_edges(node, data=True)

            out_amounts = [data.get('amount', 0.0) for _, _, data in out_edges]
            in_amounts = [data.get('amount', 0.0) for _, _, data in in_edges]

            node_features.extend([
                np.log1p(np.sum(out_amounts)),
                np.log1p(np.mean(out_amounts)) if out_amounts else 0.0,
                np.log1p(np.std(out_amounts)) if out_amounts else 0.0,
                np.log1p(np.sum(in_amounts)),
                np.log1p(np.mean(in_amounts)) if in_amounts else 0.0,
                np.log1p(np.std(in_amounts)) if in_amounts else 0.0
            ])

            # 3. Network centrality (if available)
            node_features.append(node_data.get('risk_score', 0.0))
            node_features.append(float(node_data.get('is_suspicious', False)))

            # 4. Account type features
            node_type = node_data.get('node_type', 'customer')
            type_features = [
                1.0 if node_type == 'merchant' else 0.0,
                1.0 if node_type == 'shell_company' else 0.0,
                1.0 if node_type == 'foreign' else 0.0,
                1.0 if node_type == 'crypto_exchange' else 0.0,
            ]

            node_features.extend(type_features)

            # 5. Balance features
            balance = node_data.get('balance', 0.0)
            total_sent = node_data.get('total_sent', 0.0)
            total_received = node_data.get('total_received', 0.0)

            node_features.extend([
                np.log1p(abs(balance)),
                np.log1p(total_sent),
                np.log1p(total_received)
            ])

            features.append(node_features[:20])  # Limit to 20 features

        return np.array(features, dtype=np.float32)

    def fit(self, graph: nx.DiGraph, nodes: list, labels: np.ndarray) -> None:
        """
        Train Random Forest classifier

        Args:
            graph: Transaction graph
            nodes: Training node IDs
            labels: Binary labels (1 = fraud)
        """
        print("Extracting features for Random Forest...")
        X = self.extract_features(graph, nodes)

        print("Normalizing features...")
        X = self.scaler.fit_transform(X)

        print(f"Training Random Forest on {len(nodes)} samples...")
        self.model.fit(X, labels)

        print("Training complete!")

    def predict(self, graph: nx.DiGraph, nodes: list) -> np.ndarray:
        """Predict fraud labels"""
        X = self.extract_features(graph, nodes)
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, graph: nx.DiGraph, nodes: list) -> np.ndarray:
        """Predict fraud probabilities"""
        X = self.extract_features(graph, nodes)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]


class StaticGNNBaseline:
    """
    Static GNN for node classification.

    Uses GraphSAGE or GAT for supervised node classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        gnn_type: str = "sage",
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            num_classes: Number of output classes
            gnn_type: Type of GNN ("sage" or "gat")
            num_layers: Number of GNN layers
            dropout: Dropout rate
            lr: Learning rate
            device: Training device
        """
        self.device = device
        self.gnn_type = gnn_type

        # Build GNN model
        if gnn_type == "sage":
            self.model = SAGEGNNClassifier(
                in_channels, hidden_channels, num_classes, num_layers, dropout
            ).to(device)
        elif gnn_type == "gat":
            self.model = GATGNNClassifier(
                in_channels, hidden_channels, num_classes, num_layers, dropout
            ).to(device)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)

    def networkx_to_pyg(
        self,
        graph: nx.DiGraph,
        nodes: list,
        labels: np.ndarray | None = None,
        node_feature_extractor: callable = None
    ) -> Data:
        """Convert NetworkX graph to PyG Data"""
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Extract node features
        if node_feature_extractor:
            node_features = [node_feature_extractor(graph.nodes[node]) for node in nodes]
        else:
            # Default feature extraction
            node_features = []
            for node in nodes:
                features = [
                    float(graph.out_degree(node)),
                    float(graph.in_degree(node)),
                    np.log1p(graph.nodes[node].get('total_sent', 0.0)),
                    np.log1p(graph.nodes[node].get('total_received', 0.0))
                ]
                node_features.append(features)

        x = torch.FloatTensor(node_features)

        # Extract edges (only between nodes in the list)
        edge_list = []
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_list.append([node_to_idx[u], node_to_idx[v]])

        edge_index = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.LongTensor([[], []])

        # Labels
        y = torch.LongTensor(labels) if labels is not None else None

        return Data(x=x, edge_index=edge_index, y=y)

    def fit(
        self,
        graph: nx.DiGraph,
        train_nodes: list,
        train_labels: np.ndarray,
        val_nodes: list = None,
        val_labels: np.ndarray = None,
        epochs: int = 200,
        node_feature_extractor: callable = None
    ) -> dict[str, list[float]]:
        """
        Train GNN classifier

        Args:
            graph: Transaction graph
            train_nodes: Training node IDs
            train_labels: Training labels
            val_nodes: Validation node IDs
            val_labels: Validation labels
            epochs: Number of training epochs
            node_feature_extractor: Function to extract node features

        Returns:
            Training history
        """
        print(f"Training {self.gnn_type.upper()} GNN...")

        # Convert to PyG format
        train_data = self.networkx_to_pyg(graph, train_nodes, train_labels, node_feature_extractor)
        train_data = train_data.to(self.device)

        if val_nodes is not None:
            val_data = self.networkx_to_pyg(graph, val_nodes, val_labels, node_feature_extractor)
            val_data = val_data.to(self.device)

        # Training loop
        history = {'train_loss': [], 'val_f1': []}

        for epoch in range(epochs):
            # Train
            self.model.train()
            self.optimizer.zero_grad()

            out = self.model(train_data.x, train_data.edge_index)
            loss = F.cross_entropy(out, train_data.y)

            loss.backward()
            self.optimizer.step()

            history['train_loss'].append(loss.item())

            # Validation
            if val_nodes is not None and epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(val_data.x, val_data.edge_index)
                    val_pred = val_out.argmax(dim=1).cpu().numpy()
                    val_f1 = f1_score(val_labels, val_pred, average='binary')
                    history['val_f1'].append(val_f1)

                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f}")

        print("Training complete!")
        return history

    def predict(
        self,
        graph: nx.DiGraph,
        nodes: list,
        node_feature_extractor: callable = None
    ) -> np.ndarray:
        """Predict fraud labels"""
        self.model.eval()

        data = self.networkx_to_pyg(graph, nodes, node_feature_extractor=node_feature_extractor)
        data = data.to(self.device)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1).cpu().numpy()

        return pred


class SAGEGNNClassifier(torch.nn.Module):
    """GraphSAGE classifier"""

    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class GATGNNClassifier(torch.nn.Module):
    """GAT classifier"""

    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=4, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout))

        self.convs.append(GATConv(hidden_channels * 4, num_classes, heads=1, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x
