"""
Script to generate synthetic AML dataset for prototyping
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_money_laundering.data_generator import AMLDataGenerator

if __name__ == "__main__":
    print("=" * 60)
    print("AML Synthetic Dataset Generator")
    print("=" * 60)

    # Generate small dataset for prototyping
    generator = AMLDataGenerator(
        num_accounts=200,
        num_transactions=5000,
        fraud_rate=0.02,  # 2% fraud rate (realistic)
        seed=42
    )

    print("\nGenerating transaction graph...")
    graph, accounts, transactions = generator.generate()

    print("\nSaving to files...")
    generator.save_to_files("data")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
