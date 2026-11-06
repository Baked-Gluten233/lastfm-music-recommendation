"""Summarize only the selected learning rate and three real seeds per family."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from musicrec.data import save_json


def summarize(root=ROOT):
    root=Path(root); out=root/'outputs/phase2'
    runs=pd.read_csv(out/'all_runs.csv')
    selected=json.loads((out/'selected_configs.json').read_text())
    rows=[];groups=[]
    metrics=['ndcg@10','recall@10','precision@10','hit_rate@10','catalogue_coverage@10','ndcg@20']
    for choice in selected:
        chosen=runs[(runs.family==choice['family']) & np.isclose(runs.lr,choice['lr'])]
        if set(chosen.seed)!={42,43,44} or len(chosen)!=3:
            raise ValueError('Need exactly three real seeds for selected config')
        row={'model':choice['family'],'lr':choice['lr'],'seeds':3,
             'parameters':int(chosen.parameters.iloc[0]),
             'mean_training_seconds':float(chosen.training_seconds.mean()),
             'mean_scoring_seconds':float(chosen.scoring_seconds.mean()),
             'mean_search_seconds':float(chosen.total_seconds.mean())}
        for m in metrics:
            row[m]=float(chosen[m].mean());row[m+'_seed_sd']=float(chosen[m].std(ddof=1))
        rows.append(row)
        for _,run in chosen.iterrows():
            detail=pd.read_csv(out/f"{run['run']}_users.csv")
            for label,mask in [('all',np.ones(len(detail),bool)),('has_cold_target',detail.n_cold_targets>0),
                               ('all_targets_warm',detail.n_cold_targets==0),('train_degree_below_30',detail.train_degree<30)]:
                d=detail.loc[mask]
                groups.append({'model':choice['family'],'seed':int(run.seed),'group':label,'users':len(d),
                               'ndcg@10':float(d['ndcg@10'].mean()),'recall@10':float(d['recall@10'].mean())})
    neural=pd.DataFrame(rows);neural.to_csv(out/'selected_seed_summary.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'selected_user_groups.csv',index=False)
    base=pd.read_csv(root/'outputs/validation_results.csv')
    cf=json.loads((root/'outputs/selection.json').read_text())['selected_itemcf_baseline']
    base=base[base.model.isin(['popularity',cf])]
    comparison=pd.concat([base[['model']+metrics].assign(seeds=1),neural],ignore_index=True)
    comparison.to_csv(out/'five_model_validation_comparison.csv',index=False)
    winner=comparison.sort_values('ndcg@10',ascending=False).iloc[0]
    save_json(out/'validation_recommendation.json',{'current_validation_leader':winner.model,
        'ndcg10':float(winner['ndcg@10']),'criterion':'same split, full-catalogue validation macro NDCG@10; stochastic models use chosen-config mean over 3 seeds',
        'final_test_frozen':False,'test_metrics_evaluated':False,
        'caution':'small hyperparameter budget; validation-selected checkpoints and learning rates; not independent test claims',
        'total_training_search_wall_seconds_sum':float(runs.total_seconds.sum())})
    os.environ.setdefault('MPLCONFIGDIR',str(root/'outputs/figures/.mplconfig'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'outputs/figures/.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),dpi=130)
    for col,choice in enumerate(selected):
        for _,run in runs[(runs.family==choice['family']) & (runs.seed==42)].iterrows():
            history=pd.read_csv(out/f"{run['run']}_history.csv")
            axes[0,col].plot(history.epoch,history.data_loss,label=f"lr={run.lr:g}")
            valid=history.dropna(subset=['validation_ndcg10'])
            axes[1,col].plot(valid.epoch,valid.validation_ndcg10,marker='o',label=f"lr={run.lr:g}")
        axes[0,col].set_title(choice['family']);axes[0,col].set_ylabel('Training data loss')
        axes[0,col].legend(fontsize=8)
        axes[1,col].axhline(float(base.loc[base.model==cf,'ndcg@10'].iloc[0]),color='#777777',ls='--',label='selected ItemCF')
        axes[1,col].set(xlabel='Epoch',ylabel='Validation NDCG@10',ylim=(0,.17))
        axes[1,col].legend(fontsize=8)
    fig.suptitle('All tuning candidates (seed 42): loss and full-catalogue validation',fontsize=13)
    fig.tight_layout();fig.savefig(root/'outputs/figures/04_learning_curves.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),dpi=130)
    labels=['Popularity','ItemCF','BPR-MF','NeuMF','Content two-tower']
    errors=comparison.get('ndcg@10_seed_sd',pd.Series(0,index=comparison.index)).fillna(0)
    ax.bar(labels,comparison['ndcg@10'],yerr=errors,capsize=4,color=['#999999','#31688e','#35a78d','#7755a3','#d48844'])
    ax.set(ylabel='Validation macro NDCG@10',title='Five models, same split and full-catalogue candidates')
    ax.text(.01,.98,'Error bars: SD across 3 validation-selected seed runs; not test confidence intervals',transform=ax.transAxes,va='top',fontsize=8)
    ax.set_ylim(0,.18);fig.tight_layout();fig.savefig(root/'outputs/figures/05_validation_comparison.png');plt.close(fig)
    print(comparison[['model','ndcg@10','recall@10','catalogue_coverage@10','seeds']].to_string(index=False))
    return comparison

if __name__=='__main__':summarize()
