"""
Test script for AML Detection Environment
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_money_laundering.data_generator import AMLDataGenerator
from rl_money_laundering.environment import AMLDetectionEnv, AMLDetectionEnvWrapper


def random_policy(obs: np.ndarray) -> int:
    """Simple random policy for testing"""
    return np.random.randint(0, 6)  # 5 neighbors + FLAG


def greedy_suspicious_policy(env: AMLDetectionEnv) -> int:
    """
    Greedy policy: follow most suspicious-looking edges,
    flag when fraud signal is strong
    """
    # If we've seen fraud and visited >5 nodes, flag
    fraud_edges = sum(
        1 for edge in env.visited_edges
        if env.graph.get_edge_data(*edge).get("is_fraud", False)
    )

    if fraud_edges > 0 and len(env.visited_nodes) >= 5:
        return env.FLAG_ACTION

    # Otherwise, follow most suspicious neighbor
    if not env.current_neighbors:
        return env.FLAG_ACTION

    # Score each neighbor by risk
    neighbor_scores = []
    for neighbor in env.current_neighbors:
        node_data = env.graph.nodes[neighbor]
        edge_data = env.graph.get_edge_data(env.current_node, neighbor)

        risk_score = node_data.get("risk_score", 0.0)
        is_suspicious = node_data.get("is_suspicious", False)
        is_fraud_edge = edge_data.get("is_fraud", False)

        score = risk_score + (0.5 if is_suspicious else 0.0) + (1.0 if is_fraud_edge else 0.0)
        neighbor_scores.append(score)

    # Select highest scoring neighbor
    best_idx = int(np.argmax(neighbor_scores))
    return best_idx


def main():
    print("=" * 60)
    print("AML Detection Environment Test")
    print("=" * 60)

    # Generate dataset
    print("\n1. Generating synthetic transaction graph...")
    generator = AMLDataGenerator(
        num_accounts=200,
        num_transactions=5000,
        fraud_rate=0.02,
        seed=42
    )
    graph, accounts, transactions = generator.generate()

    fraud_count = sum(1 for t in transactions if t.is_fraud)
    print(f"   Generated {len(accounts)} accounts, {len(transactions)} transactions")
    print(f"   Fraud transactions: {fraud_count} ({fraud_count/len(transactions)*100:.1f}%)")

    # Create environment
    print("\n2. Creating RL environment...")
    env = AMLDetectionEnv(
        graph=graph,
        max_steps=20,
        max_neighbors=5,
        render_mode="human"
    )
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Test random policy
    print("\n3. Testing with random policy...")
    wrapper = AMLDetectionEnvWrapper(env)

    for i in range(5):
        print(f"\n--- Episode {i+1} (Random Policy) ---")
        stats = wrapper.run_episode(lambda obs: random_policy(obs))
        print(f"Episode Reward: {stats['episode_reward']:.2f}")
        print(f"Episode Length: {stats['episode_length']}")
        print(f"Fraud Edges Found: {stats['fraud_edges_found']}")
        print(f"Fraud Detection Rate: {stats['fraud_rate']*100:.1f}%")

    random_stats = wrapper.get_statistics()
    print("\nRandom Policy Statistics (5 episodes):")
    for key, value in random_stats.items():
        print(f"  {key}: {value:.3f}")

    # Test greedy policy
    print("\n4. Testing with greedy suspicious policy...")
    wrapper2 = AMLDetectionEnvWrapper(env)

    for i in range(5):
        print(f"\n--- Episode {i+1} (Greedy Policy) ---")
        stats = wrapper2.run_episode(lambda obs: greedy_suspicious_policy(env))
        print(f"Episode Reward: {stats['episode_reward']:.2f}")
        print(f"Episode Length: {stats['episode_length']}")
        print(f"Fraud Edges Found: {stats['fraud_edges_found']}")
        print(f"Fraud Detection Rate: {stats['fraud_rate']*100:.1f}%")

    greedy_stats = wrapper2.get_statistics()
    print("\nGreedy Policy Statistics (5 episodes):")
    for key, value in greedy_stats.items():
        print(f"  {key}: {value:.3f}")

    # Compare
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Random Policy Mean Reward: {random_stats['mean_reward']:.3f}")
    print(f"Greedy Policy Mean Reward: {greedy_stats['mean_reward']:.3f}")
    print(f"Improvement: {(greedy_stats['mean_reward'] - random_stats['mean_reward']):.3f}")

    print("\n✓ Environment test complete!")


if __name__ == "__main__":
    main()
