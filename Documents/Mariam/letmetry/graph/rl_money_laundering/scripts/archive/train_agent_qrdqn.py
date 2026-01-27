"""
Compatibility wrapper for QR-DQN training.

This script is kept for backward compatibility. It forwards all CLI arguments
to ``train_agent.py`` while forcing ``--agent-type qrdqn`` so that the unified
entrypoint handles the rest of the configuration.

Prefer using:

    python scripts/train_agent.py --agent-type qrdqn ...
"""

from __future__ import annotations

import sys

from train_agent import main as _train_agent_main  # type: ignore


def _ensure_qrdqn(argv: list[str]) -> bool:
    """Force --agent-type qrdqn within the provided argv list."""
    help_requested = any(flag in argv for flag in ("-h", "--help"))

    for idx, arg in enumerate(argv):
        if arg == "--agent-type":
            if idx + 1 < len(argv):
                argv[idx + 1] = "qrdqn"
            else:
                argv.append("qrdqn")
            break
        if arg.startswith("--agent-type="):
            argv[idx] = "--agent-type=qrdqn"
            break
    else:
        argv.extend(["--agent-type", "qrdqn"])

    return help_requested


def main() -> None:
    help_requested = _ensure_qrdqn(sys.argv)
    if not help_requested:
        print(
            "Note: train_agent_qrdqn.py is deprecated. "
            "Delegating to train_agent.py with --agent-type qrdqn."
        )
    _train_agent_main()


if __name__ == "__main__":
    main()
