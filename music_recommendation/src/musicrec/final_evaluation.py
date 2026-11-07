"""Frozen final refit and one held-out evaluation round. No model selection here."""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import save_npz,load_npz
from torch.nn import functional as F

from .content import build_tag_features
from .data import indexed_matrix,save_json,sha256
from .evaluation import evaluate,bootstrap_mean_ci
from .models import popularity_scores,item_cosine,retain_neighbors,topk_unseen
from .neural import sample_negatives,score_catalogue
from .phase2 import make_model


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def verify_freeze(root):
    output=root/'outputs/final'
    decision_path=output/'frozen_decision.json'
    decision=json.loads(decision_path.read_text(encoding="utf-8"))
    receipt=json.loads((output/'freeze_receipt.json').read_text(encoding="utf-8"))
    if sha256(decision_path)!=receipt['decision_sha256']:
        raise ValueError('Frozen decision has changed')
    for filename,digest in decision['evidence_sha256'].items():
        if sha256(root/filename)!=digest:
            raise ValueError('Validation evidence changed after freeze: '+filename)
    return decision,receipt


def prepare_known_history(root):
    """Read train and validation only. No test-file path in this function."""
    manifest=json.loads((root/'outputs/split_manifest.json').read_text(encoding="utf-8"))
    frames=[]
    for split in ('train','validation'):
        path=root/f'data/processed/{split}.csv'
        if sha256(path)!=manifest['split_sha256'][split]:
            raise ValueError('Frozen development split changed')
        frames.append(pd.read_csv(path))
    known=pd.concat(frames,ignore_index=True).sort_values(['userID','artistID']).reset_index(drop=True)
    if known.duplicated(['userID','artistID']).any():
        raise ValueError('Known-history duplicate pair')
    audit=json.loads((root/'outputs/data_audit.json').read_text(encoding="utf-8"))
    raw=root/'data/raw/hetrec2011-lastfm-2k'
    for filename in ('artists.dat','user_taggedartists-timestamps.dat'):
        if sha256(raw/filename)!=audit['raw_files_sha256'][filename]:
            raise ValueError('Raw source changed: '+filename)
    artists=pd.read_csv(raw/'artists.dat',sep='\t')
    events=pd.read_csv(raw/'user_taggedartists-timestamps.dat',sep='\t')
    matrix,user_ids,artist_ids,user_index,item_index=indexed_matrix(known,artists)
    feature_ids,feature_weights,vocabulary,content_audit=build_tag_features(events,known,artist_ids,
                                                            min_df=2,max_features=2000,max_tags=32)
    content_audit['eligibility']='tag (userID,artistID) must occur in train.csv + validation.csv; test pairs excluded'
    content_audit['fit_data']='train + validation known history'
    pop=popularity_scores(matrix)
    return {'known':known,'artists':artists,'matrix':matrix,'user_ids':user_ids,'artist_ids':artist_ids,
            'user_index':user_index,'item_index':item_index,'popularity':pop,'warm':pop>0,
            'seen':matrix.toarray().astype(bool),'users':known.userID.map(user_index).to_numpy(dtype=np.int64),
            'items':known.artistID.map(item_index).to_numpy(dtype=np.int64),
            'feature_ids':feature_ids,'feature_weights':feature_weights,'vocabulary':vocabulary,
            'content_audit':content_audit}


def fit_fixed_epochs(config,data,output,seed):
    """Fixed-budget training, no validation callback and no early stopping."""
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed)
    model=make_model(config,data)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=1e-4)
    name=f"{config['family']}_seed{seed}"
    history=[];start=time.perf_counter()
    for epoch in range(1,config['final_epochs']+1):
        negative=sample_negatives(data['users'],data['seen'],np.flatnonzero(data['warm']),rng)
        order=rng.permutation(len(data['users']));loss_sum=0.;count=0
        model.train()
        for beginning in range(0,len(order),1024):
            select=order[beginning:beginning+1024]
            users=torch.from_numpy(data['users'][select])
            positives=torch.from_numpy(data['items'][select]);negatives=torch.from_numpy(negative[select])
            optimizer.zero_grad(set_to_none=True)
            p=model(users,positives);n=model(users,negatives)
            loss=F.softplus(n-p).mean() if model.objective=='bpr' else (F.softplus(-p).mean()+F.softplus(n).mean())/2
            if not torch.isfinite(loss):raise ValueError('Nonfinite final-training loss')
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(select);count+=len(select)
        history.append({'epoch':epoch,'data_loss':loss_sum/count})
    seconds=time.perf_counter()-start
    pd.DataFrame(history).to_csv(output/f'{name}_refit_history.csv',index=False)
    checkpoint=output/f'{name}_weights.npz'
    np.savez_compressed(checkpoint,**{key:value.detach().numpy() for key,value in model.state_dict().items()})
    info={'run':name,'family':config['family'],'seed':seed,'dim':config['dim'],'lr':config['lr'],
          'epochs':config['final_epochs'],'fit_seconds':seconds,'parameters':sum(p.numel() for p in model.parameters()),
          'checkpoint_sha256':sha256(checkpoint),'saved_at_utc':utcnow()}
    print(f"Final refit saved: {name}, fixed epochs={config['final_epochs']}, train={seconds:.2f}s",flush=True)
    return info


def make_cases(data,test,similarity,decision):
    names=data['artists'].set_index('id')['name'].to_dict();cases=[]
    for uid in decision['case_user_ids']:
        user=data['user_index'][uid]
        seen=data['matrix'].indices[data['matrix'].indptr[user]:data['matrix'].indptr[user+1]]
        scores=(data['matrix'].getrow(user)@similarity).toarray().ravel()
        ranked=topk_unseen(scores,seen,data['popularity'],10)
        relevant=set(test.loc[test.userID==uid,'artistID'].astype(int))
        known=data['known'][data['known'].userID==uid].sort_values(['weight','artistID'],ascending=[False,True]).head(5)
        recommendations=[]
        for rank,column in enumerate(ranked,1):
            aid=int(data['artist_ids'][column]);contributions=similarity[seen,column].toarray().ravel()
            if len(contributions) and contributions.max()>0:
                origin=int(data['artist_ids'][seen[np.argmax(contributions)]])
                reason={'largest_known_history_contributor_id':origin,'name':str(names[origin]),
                        'similarity_contribution':float(contributions.max())}
            else:reason={'fallback':'zero similarity score; known-history popularity tie-break'}
            recommendations.append({'rank':rank,'artistID':aid,'name':str(names[aid]),'score':float(scores[column]),
                                    'test_hit':aid in relevant,'explanation':reason})
        cases.append({'userID':uid,'selection_rule':'first three validation user IDs, fixed before test was read',
                      'known_history_examples':[{'artistID':int(r.artistID),'name':str(names[int(r.artistID)]),'listening_count':int(r.weight)} for r in known.itertuples()],
                      'test_artists':[{'artistID':aid,'name':str(names[aid])} for aid in sorted(relevant)],
                      'recommendations':recommendations})
    return cases


def run_final_once(root):
    root=Path(root).resolve();output=root/'outputs/final'
    decision,receipt=verify_freeze(root)
    state_path=output/'evaluation_state.json'
    if state_path.exists():
        raise RuntimeError('Final evaluation state already exists. Refusing an automatic second final pass; inspect the saved state and results.')
    torch.set_num_threads(decision['threads'])
    state={'status':'refitting','decision_sha256':receipt['decision_sha256'],'refit_started_at_utc':utcnow(),
           'test_labels_read':False,'test_metrics_evaluated':False}
    save_json(state_path,state)
    data=prepare_known_history(root)
    save_json(output/'refit_content_audit.json',data['content_audit'])
    save_json(output/'refit_tag_vocabulary.json',data['vocabulary'])
    np.savez_compressed(output/'refit_content_features.npz',ids=data['feature_ids'],weights=data['feature_weights'])
    save_npz(output/'known_history_matrix.npz',data['matrix'])
    np.save(output/'popularity_scores.npy',data['popularity'])
    fit_start=time.perf_counter()
    similarity=retain_neighbors(item_cosine(data['matrix'],10.0),100)
    cf_seconds=time.perf_counter()-fit_start
    save_npz(output/'itemcf_similarity.npz',similarity)
    manifest=[{'run':'popularity','family':'popularity','seed':None,'fit_seconds':0.,'checkpoint_sha256':sha256(output/'popularity_scores.npy'),'saved_at_utc':utcnow()},
              {'run':'itemcf','family':'itemcf','seed':None,'fit_seconds':cf_seconds,'checkpoint_sha256':sha256(output/'itemcf_similarity.npz'),'saved_at_utc':utcnow()}]
    for config in decision['neural_models']:
        for seed in config['seeds']:
            manifest.append(fit_fixed_epochs(config,data,output,seed))
    save_json(output/'final_checkpoint_manifest.json',manifest)
    state.update({'status':'all_frozen_models_saved','all_checkpoints_saved_at_utc':utcnow(),
                  'checkpoint_count':len(manifest),'known_history_rows':len(data['known']),
                  'known_history_users':len(data['user_ids']),'known_history_artists':int(data['warm'].sum())})
    save_json(state_path,state)
    # No final code above this line reads the test file. All learned state exists.
    verify_freeze(root)
    state.update({'status':'evaluating','test_evaluation_started_at_utc':utcnow()})
    save_json(state_path,state)
    test_path=root/'data/processed/test.csv'
    if sha256(test_path)!=decision['test_split_sha256']:raise ValueError('Held-out test hash changed')
    test=pd.read_csv(test_path)
    state['test_labels_read']=True;save_json(state_path,state)
    results=[];details={}
    for item in manifest:
        scoring_start=time.perf_counter()
        if item['family']=='popularity':scores=None
        elif item['family']=='itemcf':scores=data['matrix']@similarity
        else:
            config=next(c for c in decision['neural_models'] if c['family']==item['family'])
            model=make_model(config,data)
            with np.load(output/f"{item['run']}_weights.npz",allow_pickle=False) as weights:
                model.load_state_dict({key:torch.from_numpy(weights[key].copy()) for key in weights.files},strict=True)
            scores=score_catalogue(model,*data['matrix'].shape,data['warm'])
        scoring_seconds=time.perf_counter()-scoring_start
        metric_start=time.perf_counter()
        metrics,detail=evaluate(data['matrix'],test,data['user_index'],data['item_index'],data['popularity'],scores)
        details[item['run']]=detail
        detail.to_csv(output/f"{item['run']}_test_users.csv",index=False)
        results.append({**item,**metrics,'scoring_seconds':scoring_seconds,'metric_seconds':time.perf_counter()-metric_start})
        pd.DataFrame(results).to_csv(output/'all_test_runs.csv',index=False)
        print(f"Frozen test result: {item['run']}, NDCG@10={metrics['ndcg@10']:.6f}",flush=True)
    runs=pd.DataFrame(results);rows=[]
    metric_names=['ndcg@10','recall@10','precision@10','hit_rate@10','catalogue_coverage@10','ndcg@20','recall@20','precision@20','hit_rate@20']
    for family in decision['comparison_families']:
        selected=runs[runs.family==family]
        row={'model':family,'seeds':len(selected),'frozen_primary':family=='itemcf','mean_fit_seconds':float(selected.fit_seconds.mean()),
             'mean_scoring_seconds':float(selected.scoring_seconds.mean())}
        for name in metric_names:
            row[name]=float(selected[name].mean());row[name+'_seed_sd']=float(selected[name].std(ddof=1)) if len(selected)>1 else 0.
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'five_model_test_comparison.csv',index=False)
    groups=[]
    for item in manifest:
        detail=details[item['run']]
        for label,mask in [('all',np.ones(len(detail),bool)),('has_cold_target',detail.n_cold_targets>0),
                           ('all_targets_warm',detail.n_cold_targets==0),('known_history_degree_below_30',detail.train_degree<30)]:
            subset=detail.loc[mask]
            groups.append({'run':item['run'],'family':item['family'],'seed':item['seed'],'group':label,'users':len(subset),
                           'ndcg@10':float(subset['ndcg@10'].mean()) if len(subset) else None,
                           'recall@10':float(subset['recall@10'].mean()) if len(subset) else None})
    pd.DataFrame(groups).to_csv(output/'test_user_groups.csv',index=False)
    paired=details['itemcf'].merge(details['popularity'],on='userID',suffixes=('_cf','_pop'),validate='one_to_one')
    save_json(output/'primary_paired_user_bootstrap.json',{
        'contrast':'frozen ItemCF minus Popularity, per-user NDCG@10',
        **bootstrap_mean_ci(paired['ndcg@10_cf']-paired['ndcg@10_pop']),
        'limitation':'conditional on this split and these trained models; not split uncertainty, not an online causal effect'})
    save_json(output/'representative_cases.json',make_cases(data,test,similarity,decision))
    state.update({'status':'completed','test_metrics_evaluated':True,'completed_at_utc':utcnow(),
                  'evaluated_runs':len(results),'test_users':int(test.userID.nunique()),'test_interactions':len(test),
                  'test_cold_interactions':int((~test.artistID.isin(data['known'].artistID)).sum()),
                  'primary_model_unchanged':'itemcf_n100_s10','test_used_for_tuning':False,
                  'windows_execution_verified':platform.system()=='Windows','torch':torch.__version__})
    save_json(state_path,state)
    return pd.DataFrame(rows)
