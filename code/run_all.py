"""Run every generator and plot every available figure."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=HERE.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    args = parser.parse_args()
    run([sys.executable, str(HERE / "verify_theory.py")])
    run(
        [
            sys.executable,
            str(HERE / "generate_data.py"),
            "--experiment",
            "all",
            "--profile",
            args.profile,
        ]
    )
    run([sys.executable, str(HERE / "plot_results.py"), "--figure", "all"])


if __name__ == "__main__":
    main()
