import torch

from rl_money_laundering.gnn_encoder import TGATEncoder, TGNEncoder


def _toy_inputs(num_nodes: int = 4, feature_dim: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
    edge_time = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32)
    return x, edge_index, edge_time


def test_tgat_encoder_produces_expected_shape() -> None:
    torch.set_num_threads(1)
    x, edge_index, edge_time = _toy_inputs()
    encoder = TGATEncoder(
        in_channels=x.size(1),
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        heads=2,
        time_dim=16,
    )
    output = encoder(x, edge_index, edge_time)
    assert output.shape == (x.size(0), 16)


def test_tgn_encoder_produces_expected_shape() -> None:
    torch.set_num_threads(1)
    x, edge_index, edge_time = _toy_inputs()
    encoder = TGNEncoder(
        in_channels=x.size(1),
        hidden_channels=32,
        out_channels=16,
        time_dim=16,
    )
    output = encoder(x, edge_index, edge_time)
    assert output.shape == (x.size(0), 16)
