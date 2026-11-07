"""Recompute validation metrics from safe NPZ checkpoints; never read test.csv."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from musicrec.phase2 import prepare,make_model,evaluate_model
from musicrec.data import save_json
from musicrec.validation import require


def replay(root=ROOT):
    root=Path(root);output=root/'outputs/phase2';torch.set_num_threads(4)
    data=prepare(root)
    selected=json.loads((output/'selected_configs.json').read_text(encoding="utf-8"))
    all_runs=pd.read_csv(output/'all_runs.csv')
    if {c['family'] for c in selected}!={'bpr_mf','neumf','content_two_tower'} or len(selected)!=3:
        raise AssertionError('Replay requires exactly the three selected model families')
    checks=[]
    for config in selected:
        runs=all_runs[(all_runs.family==config['family']) & np.isclose(all_runs.lr,config['lr'])]
        if len(runs)!=3 or set(runs.seed)!={42,43,44}:
            raise AssertionError('Replay requires three distinct selected-config seeds per family')
        for _,row in runs.iterrows():
            model=make_model(config,data)
            with np.load(output/f"{row['run']}_weights.npz",allow_pickle=False) as weights:
                state={k:torch.from_numpy(weights[k].copy()) for k in weights.files}
            model.load_state_dict(state,strict=True)
            metrics,_=evaluate_model(model,data)
            keys=['ndcg@10','recall@10','precision@10','ndcg@20','catalogue_coverage@10']
            differences={k:abs(float(row[k])-metrics[k]) for k in keys}
            if not all(np.isfinite(value) for value in differences.values()) or max(differences.values())>1e-7:
                raise AssertionError(f"Checkpoint replay mismatch: {row['run']} {differences}")
            checks.append({'run':row['run'],'max_abs_metric_error':max(differences.values()),
                           'ndcg@10':metrics['ndcg@10'],'passed':True})
            print('Replayed:',row['run'],f"NDCG@10={metrics['ndcg@10']:.6f}",flush=True)
    require(len(checks)==9, 'Consistency check failed: len(checks)==9')
    save_json(output/'checkpoint_replay.json',{'checks':checks,'test_metrics_evaluated':False})
    return pd.DataFrame(checks)

if __name__=='__main__':replay()
