"""Audit user-authored tags, optionally run the isolated validation comparison."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from musicrec.content_feature_audit import run_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run-validation', action='store_true')
    mode.add_argument('--replay-only', action='store_true')
    args = parser.parse_args()
    result = run_audit(args.root, args.output_dir, args.run_validation, args.replay_only)
    print(result.to_string(index=False) if hasattr(result, 'to_string') else result)


if __name__ == '__main__':
    main()
