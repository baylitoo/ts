import networkx as nx
from rl_money_laundering.environment import AMLDetectionEnv


def _build_test_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    default_node_attrs = {
        "account_type": "individual",
        "balance": 0.0,
        "risk_score": 0.0,
        "is_suspicious": False,
        "first_step": 0,
        "last_step": 0,
    }
    graph.add_node("clean", **default_node_attrs)
    graph.add_node("fraud", **{**default_node_attrs, "risk_score": 1.0})
    graph.add_edge("clean", "fraud", amount=1000.0, step=0, is_fraud=1)
    graph.add_edge("fraud", "clean", amount=500.0, step=1)  # keep environment from terminating immediately
    return graph


def test_flag_reward_matches_correctness_true_positive() -> None:
    env = AMLDetectionEnv(_build_test_graph(), max_steps=5, intrinsic_reward_weight=0.0)
    env.reset(options={"start_node": "fraud"})

    _, reward, terminated, truncated, info = env.step(env.FLAG_ACTION, confidence=1.0)

    assert terminated is True
    assert truncated is False
    assert info["flag_used"] is True
    assert info["flag_correct"] is True
    assert reward > 0.0, "True positive flag should yield positive reward"


def test_flag_reward_matches_correctness_false_positive() -> None:
    env = AMLDetectionEnv(_build_test_graph(), max_steps=5, intrinsic_reward_weight=0.0)
    env.reset(options={"start_node": "clean"})

    _, reward, terminated, truncated, info = env.step(env.FLAG_ACTION, confidence=1.0)

    assert terminated is True
    assert truncated is False
    assert info["flag_used"] is True
    assert info["flag_correct"] is False
    assert reward < 0.0, "False positive flag should yield negative reward"
