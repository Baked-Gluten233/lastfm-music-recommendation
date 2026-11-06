from pathlib import Path
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from musicrec.final_evaluation import run_final_once

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='One frozen train+validation refit and final test round')
    parser.add_argument('--execute-frozen-test',action='store_true',required=True)
    parser.parse_args()
    run_final_once(ROOT)
