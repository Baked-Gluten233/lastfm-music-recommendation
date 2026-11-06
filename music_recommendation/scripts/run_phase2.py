from pathlib import Path
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from musicrec.phase2 import run_phase2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Phase2 development: frozen train/validation only')
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--every',type=int,default=4)
    parser.add_argument('--skip-repeat-seeds',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 or args.every<1: parser.error('epochs/every must be positive')
    run_phase2(ROOT,not args.skip_repeat_seeds,args.epochs,args.every)
