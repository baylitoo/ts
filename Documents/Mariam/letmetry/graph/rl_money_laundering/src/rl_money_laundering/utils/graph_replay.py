"""Graph-aware replay buffer for GNN training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, Data


@dataclass
class GraphReplayBatch:
    graphs: Batch
    next_graphs: Batch
    histories: torch.Tensor
    next_histories: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    is_fraud: torch.Tensor


class GraphReplayBuffer:
    """Replay buffer that stores graph-structured transitions."""

    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: Deque[Tuple[Data, np.ndarray, int, float, Data, np.ndarray, bool, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        graph_state: Data,
        history_features: np.ndarray,
        action: int,
        reward: float,
        next_graph_state: Data,
        next_history_features: np.ndarray,
        done: bool,
        is_fraud: bool = False,
    ) -> None:
        self.buffer.append(
            (
                graph_state,
                history_features,
                action,
                reward,
                next_graph_state,
                next_history_features,
                done,
                is_fraud,
            )
        )

    def sample(self, batch_size: int) -> GraphReplayBatch:
        batch = random.sample(self.buffer, batch_size)
        graphs, histories, actions, rewards, next_graphs, next_histories, dones, is_fraud = zip(*batch)

        graph_batch = Batch.from_data_list(list(graphs))
        next_graph_batch = Batch.from_data_list(list(next_graphs))

        histories_tensor = torch.tensor(np.stack(histories), dtype=torch.float32)
        next_histories_tensor = torch.tensor(np.stack(next_histories), dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        fraud_tensor = torch.tensor(is_fraud, dtype=torch.float32)

        return GraphReplayBatch(
            graphs=graph_batch,
            next_graphs=next_graph_batch,
            histories=histories_tensor,
            next_histories=next_histories_tensor,
            actions=actions_tensor,
            rewards=rewards_tensor,
            dones=dones_tensor,
            is_fraud=fraud_tensor,
        )

    def __len__(self) -> int:
        return len(self.buffer)
