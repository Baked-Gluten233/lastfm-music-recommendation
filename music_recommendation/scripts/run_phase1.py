"""Run from any current directory: python scripts/run_phase1.py."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicrec.pipeline import run_phase1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/validation only; no final test evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_phase1(ROOT, seed=args.seed)
