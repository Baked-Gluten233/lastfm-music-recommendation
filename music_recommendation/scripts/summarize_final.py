"""Verify and visualize already-saved final outputs, without rereading test.csv."""
from pathlib import Path
from datetime import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from musicrec.data import save_json,sha256
from musicrec.validation import require
from musicrec.final_evaluation import verify_freeze


def verify_and_summarize(root=ROOT):
    root=Path(root);out=root/'outputs/final'
    decision,receipt=verify_freeze(root)
    state=json.loads((out/'evaluation_state.json').read_text(encoding="utf-8"))
    if state['status']!='completed':raise ValueError('Final evaluation is not complete')
    manifest=json.loads((out/'final_checkpoint_manifest.json').read_text(encoding="utf-8"))
    require(len(manifest)==11 and state['evaluated_runs']==11, "Consistency check failed: len(manifest)==11 and state['evaluated_runs']==11")
    require(datetime.fromisoformat(state['all_checkpoints_saved_at_utc'])<datetime.fromisoformat(state['test_evaluation_started_at_utc']), "Consistency check failed: datetime.fromisoformat(state['all_checkpoints_saved_at_utc'])<datetime.fromisoformat(state['test_evaluation_started_at_utc'])")
    require(datetime.fromisoformat(decision['frozen_at_utc'])<datetime.fromisoformat(state['refit_started_at_utc']), "Consistency check failed: datetime.fromisoformat(decision['frozen_at_utc'])<datetime.fromisoformat(state['refit_started_at_utc'])")
    for checkpoint in manifest:
        name=checkpoint['run']
        file='popularity_scores.npy' if name=='popularity' else 'itemcf_similarity.npz' if name=='itemcf' else name+'_weights.npz'
        require(sha256(out/file)==checkpoint['checkpoint_sha256'], "Consistency check failed: sha256(out/file)==checkpoint['checkpoint_sha256']")
        if checkpoint['family'] not in ('popularity','itemcf'):
            frozen=next(c for c in decision['neural_models'] if c['family']==checkpoint['family'])
            require(checkpoint['epochs']==frozen['final_epochs'], "Consistency check failed: checkpoint['epochs']==frozen['final_epochs']")
            h=pd.read_csv(out/f'{name}_refit_history.csv')
            require(len(h)==frozen['final_epochs'], "Consistency check failed: len(h)==frozen['final_epochs']")
            require(h.epoch.tolist()==list(range(1,frozen['final_epochs']+1)), "Consistency check failed: h.epoch.tolist()==list(range(1,frozen['final_epochs']+1))")
        require(datetime.fromisoformat(checkpoint['saved_at_utc'])<datetime.fromisoformat(state['test_evaluation_started_at_utc']), "Consistency check failed: datetime.fromisoformat(checkpoint['saved_at_utc'])<datetime.fromisoformat(state['test_evaluation_started_at_utc'])")
    runs=pd.read_csv(out/'all_test_runs.csv')
    comparison=pd.read_csv(out/'five_model_test_comparison.csv')
    metrics=['ndcg@10','recall@10','precision@10','hit_rate@10','ndcg@20','recall@20','precision@20','hit_rate@20']
    for _,row in runs.iterrows():
        users=pd.read_csv(out/f"{row['run']}_test_users.csv")
        for metric in metrics:
            require(abs(users[metric].mean()-row[metric])<1e-12, 'Consistency check failed: abs(users[metric].mean()-row[metric])<1e-12')
        require(len(users)==int(row.evaluated_users), 'Consistency check failed: len(users)==int(row.evaluated_users)')
        require(users.n_targets.sum()==int(row.targets), 'Consistency check failed: users.n_targets.sum()==int(row.targets)')
        require(users.n_cold_targets.sum()==int(row.cold_targets), 'Consistency check failed: users.n_cold_targets.sum()==int(row.cold_targets)')
    for _,summary in comparison.iterrows():
        selected=runs[runs.family==summary.model]
        if summary.model in ('popularity','itemcf'):require(len(selected)==1, 'Consistency check failed: len(selected)==1')
        else:require(len(selected)==3 and set(selected.seed)=={42,43,44}, 'Consistency check failed: len(selected)==3 and set(selected.seed)=={42,43,44}')
        for metric in metrics+['catalogue_coverage@10']:
            require(abs(selected[metric].mean()-summary[metric])<1e-12, 'Consistency check failed: abs(selected[metric].mean()-summary[metric])<1e-12')
            sd=selected[metric].std(ddof=1) if len(selected)>1 else 0.
            require(abs(sd-summary[metric+'_seed_sd'])<1e-12, "Consistency check failed: abs(sd-summary[metric+'_seed_sd'])<1e-12")
    groups=pd.read_csv(out/'test_user_groups.csv')
    group_summary=groups.groupby(['family','group'],sort=False).agg(users=('users','first'),ndcg10_mean=('ndcg@10','mean'),recall10_mean=('recall@10','mean')).reset_index()
    group_summary.to_csv(out/'test_group_summary.csv',index=False)
    save_json(out/'verification.json',{'freeze_and_evidence_hashes':'passed','chronology':'all eleven checkpoints saved before test was read',
        'checkpoint_hashes':11,'individual_runs_aggregates_verified':11,'five_model_mean_and_sample_sd':'passed',
        'actual_training_epochs_match_frozen_values':True,'primary_model_still_frozen_itemcf':True,
        'test_labels_reread_for_this_verification':False,'test_scores_recomputed_for_this_verification':False})
    os.environ.setdefault('MPLCONFIGDIR',str(root/'outputs/figures/.mplconfig'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'outputs/figures/.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Popularity','ItemCF (frozen)','BPR-MF','NeuMF','Content two-tower']
    fig,ax=plt.subplots(figsize=(9,4.6),dpi=150)
    ax.bar(labels,comparison['ndcg@10'],yerr=comparison['ndcg@10_seed_sd'],capsize=4,
           color=['#999999','#31688e','#35a78d','#7755a3','#d48844'])
    ax.set(ylabel='Test macro NDCG@10',title='Frozen final comparison after refitting on train + validation')
    ax.text(.01,.98,'Error bars: sample SD of 3 fixed-budget seeds, not confidence intervals',transform=ax.transAxes,fontsize=9,va='top')
    ax.set_ylim(0,max(.22,float(comparison['ndcg@10'].max())*1.25))
    fig.tight_layout();fig.savefig(root/'outputs/figures/06_final_test_comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.7),dpi=150)
    colors=['#999999','#31688e','#35a78d','#7755a3','#d48844']
    for i,(_,row) in enumerate(comparison.iterrows()):
        ax.scatter(row['catalogue_coverage@10']*100,row['ndcg@10'],s=85,color=colors[i])
        offset=(5,-16) if i==2 else (5,5)
        ax.annotate(labels[i],(row['catalogue_coverage@10']*100,row['ndcg@10']),xytext=offset,textcoords='offset points',fontsize=9)
    ax.set(xlabel='Catalogue coverage@10 (%)',ylabel='Test macro NDCG@10',title='Accuracy and catalogue coverage: frozen models')
    ax.set_xlim(left=-.5,right=max(10,float(comparison['catalogue_coverage@10'].max())*100+4))
    ax.set_ylim(0,.22);fig.tight_layout();fig.savefig(root/'outputs/figures/07_test_accuracy_coverage.png');plt.close(fig)
    print(comparison[['model','ndcg@10','ndcg@10_seed_sd','recall@10','catalogue_coverage@10']].to_string(index=False))
    return comparison,group_summary

if __name__=='__main__':verify_and_summarize()
