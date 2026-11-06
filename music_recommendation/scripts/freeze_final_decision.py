"""Create an immutable pre-test decision using validation evidence only."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from musicrec.data import sha256,save_json


def freeze():
    output=ROOT/'outputs/final';output.mkdir(parents=True,exist_ok=True)
    path=output/'frozen_decision.json'
    if path.exists():
        raise RuntimeError('A final decision already exists; do not silently overwrite it')
    runs=pd.read_csv(ROOT/'outputs/phase2/all_runs.csv')
    configs=json.loads((ROOT/'outputs/phase2/selected_configs.json').read_text())
    leader=json.loads((ROOT/'outputs/phase2/validation_recommendation.json').read_text())
    if leader['current_validation_leader']!='itemcf_n100_s10':
        raise ValueError('Validation evidence differs from the reviewed final choice')
    frozen=[]
    for config in configs:
        subset=runs[(runs.family==config['family']) & np.isclose(runs.lr,config['lr'])]
        if len(subset)!=3 or set(subset.seed)!={42,43,44}:
            raise ValueError('Need the three selected-configuration validation seeds')
        epochs=subset.sort_values('seed').selected_epoch.astype(int).tolist()
        frozen.append({'family':config['family'],'dim':int(config['dim']),'lr':float(config['lr']),
                       'validation_selected_epochs':epochs,'final_epochs':int(np.median(epochs)),
                       'seeds':[42,43,44]})
    manifest=json.loads((ROOT/'outputs/split_manifest.json').read_text())
    validation=pd.read_csv(ROOT/'data/processed/validation.csv')
    decision={
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'selected_primary_model':{'family':'itemcf','neighbors':100,'shrinkage':10.0},
        'selection_evidence':'Validation macro NDCG@10: ItemCF 0.144628987; chosen neural configuration means below this',
        'comparison_families':['popularity','itemcf','bpr_mf','neumf','content_two_tower'],
        'neural_models':frozen,
        'epoch_rule':'Median of the three existing selected-configuration validation-selected epochs; chosen now before reading test labels, not specified numerically in phase1',
        'training_data':'train.csv + validation.csv, every model refitted on this same known history',
        'optimizer':'Adam','weight_decay':1e-4,'batch_size':1024,'threads':4,'device':'cpu',
        'training_negative_sampling':'one uniformly sampled train+validation-warm artist absent from that user known history per positive per epoch; never consult test',
        'content_features':{'allowed_pairs':'train + validation only','min_df':2,'max_features':2000,'max_tags':32,
                            'vocabulary_order':'descending known-history artist document frequency, then numeric tagID',
                            'idf_documents':'known-history observed artists, including empty documents','unknown_feature':True},
        'neural_cold_scores':'minimum warm score for that user minus 1; cold artists remain candidates and denominators',
        'evaluation':{'primary_metric':'user-macro NDCG@10','ks':[10,20],
                      'metrics':['precision','recall','ndcg','hit_rate','catalogue_coverage@10'],
                      'candidates':'all artists.dat IDs minus user train+validation history; no sampled evaluation negatives',
                      'tie_break':'score descending, known-history user frequency descending, original artist ID ascending',
                      'groups':['all','has_cold_target','all_targets_warm','known_history_degree_below_30'],
                      'neural_aggregation':'report all seeds and mean/sample SD ddof=1; no seed selection or score ensemble',
                      'bootstrap':'1000 user resamples, seed2026, primary ItemCF-minus-Popularity NDCG@10 contrast only'},
        'case_user_ids':sorted(map(int,validation.userID.unique()))[:3],
        'test_split_sha256':manifest['split_sha256']['test'],
        'evidence_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'outputs/phase2/all_runs.csv',ROOT/'outputs/phase2/selected_configs.json',ROOT/'outputs/split_manifest.json']},
        'test_labels_read_for_this_decision':False,'test_metrics_available_at_freeze':False,
        'test_policy':'Train and save all final checkpoints before reading test.csv. One final evaluation round for all frozen comparisons. Never retune or replace primary model from test results.',
    }
    save_json(path,decision)
    save_json(output/'freeze_receipt.json',{'decision_sha256':sha256(path),'frozen_at_utc':decision['frozen_at_utc']})
    print(json.dumps(decision,ensure_ascii=False,indent=2))

if __name__=='__main__':freeze()
