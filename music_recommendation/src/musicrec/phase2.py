"""CPU model development on frozen train/validation only; test file never read."""
from __future__ import annotations

import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .content import build_tag_features
from .data import indexed_matrix,save_json,sha256
from .evaluation import evaluate
from .models import popularity_scores
from .neural import (BPRMF,NeuMF,ContentTwoTower,CrossFittedContentTwoTower,
                     sample_negatives,score_catalogue)

GRID = [
    {'family':'bpr_mf','dim':32,'lr':.005},
    {'family':'bpr_mf','dim':32,'lr':.01},
    {'family':'neumf','dim':16,'lr':.001},
    {'family':'neumf','dim':16,'lr':.003},
    {'family':'content_two_tower','dim':32,'lr':.001},
    {'family':'content_two_tower','dim':32,'lr':.003},
]


def prepare(root):
    manifest=json.loads((root/'outputs/split_manifest.json').read_text(encoding="utf-8"))
    audit=json.loads((root/'outputs/data_audit.json').read_text(encoding="utf-8"))
    for filename in ('artists.dat','user_taggedartists-timestamps.dat'):
        if sha256(root/'data/raw/hetrec2011-lastfm-2k'/filename) != audit['raw_files_sha256'][filename]:
            raise ValueError('Original input changed: '+filename)
    for split in ('train','validation'):
        if sha256(root/f'data/processed/{split}.csv') != manifest['split_sha256'][split]:
            raise ValueError('Frozen split has changed: '+split)
    train=pd.read_csv(root/'data/processed/train.csv')
    validation=pd.read_csv(root/'data/processed/validation.csv')
    artists=pd.read_csv(root/'data/raw/hetrec2011-lastfm-2k/artists.dat',sep='\t')
    events=pd.read_csv(root/'data/raw/hetrec2011-lastfm-2k/user_taggedartists-timestamps.dat',sep='\t')
    matrix,user_ids,artist_ids,user_index,item_index=indexed_matrix(train,artists)
    feature_ids,feature_weights,vocabulary,content_audit=build_tag_features(events,train,artist_ids)
    pop=popularity_scores(matrix)
    users=train.userID.map(user_index).to_numpy(dtype=np.int64)
    items=train.artistID.map(item_index).to_numpy(dtype=np.int64)
    return {'train':train,'validation':validation,'matrix':matrix,'user_ids':user_ids,
            'artist_ids':artist_ids,'user_index':user_index,'item_index':item_index,
            'popularity':pop,'warm':pop>0,'seen':matrix.toarray().astype(bool),
            'users':users,'items':items,'feature_ids':feature_ids,'feature_weights':feature_weights,
            'vocabulary':vocabulary,'content_audit':content_audit}


def make_model(config,data):
    users,items=data['matrix'].shape
    family=config['family']; dim=config['dim']
    if family=='bpr_mf': return BPRMF(users,items,dim)
    if family=='neumf': return NeuMF(users,items,dim)
    if family=='content_two_tower':
        return ContentTwoTower(users,data['feature_ids'],data['feature_weights'],len(data['vocabulary']),dim)
    if family=='content_two_tower_audit':
        return CrossFittedContentTwoTower(users,data['feature_ids'],data['feature_weights'],
                                         len(data['vocabulary']),data['user_folds'],dim)
    raise ValueError(family)


def evaluate_model(model,data):
    score_start=time.perf_counter()
    scores=score_catalogue(model,*data['matrix'].shape,data['warm'])
    score_seconds=time.perf_counter()-score_start
    metric_start=time.perf_counter()
    metrics,detail=evaluate(data['matrix'],data['validation'],data['user_index'],
                            data['item_index'],data['popularity'],scores)
    metrics['scoring_seconds']=score_seconds
    metrics['metric_seconds']=time.perf_counter()-metric_start
    return metrics,detail


def train_config(config,data,output,seed=42,max_epochs=24,every=4,patience=3):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed)
    model=make_model(config,data)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=1e-4)
    name=f"{config['family']}_lr{config['lr']:g}_seed{seed}"
    total_start=time.perf_counter()
    best=-float('inf'); best_epoch=0; best_state=None; stale=0
    history=[]; cumulative_train=0.; cumulative_eval=0.
    warm_items=np.flatnonzero(data['warm'])
    for epoch in range(1,max_epochs+1):
        epoch_start=time.perf_counter()
        negatives=sample_negatives(data['users'],data['seen'],warm_items,rng)
        order=rng.permutation(len(data['users']))
        model.train(); loss_sum=0.; count=0
        for start in range(0,len(order),1024):
            select=order[start:start+1024]
            users=torch.from_numpy(data['users'][select])
            positives=torch.from_numpy(data['items'][select])
            negative=torch.from_numpy(negatives[select])
            optimizer.zero_grad(set_to_none=True)
            positive_score=model(users,positives)
            negative_score=model(users,negative)
            if model.objective=='bpr':
                loss=F.softplus(negative_score-positive_score).mean()
            else:
                loss=(F.softplus(-positive_score).mean()+F.softplus(negative_score).mean())/2
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            loss_sum+=float(loss.detach())*len(select); count+=len(select)
        train_seconds=time.perf_counter()-epoch_start
        cumulative_train+=train_seconds
        row={'epoch':epoch,'training_objective':model.objective,'data_loss':loss_sum/count,
             'train_seconds':train_seconds,'validation_ndcg10':None,'validation_recall10':None}
        if epoch % every == 0 or epoch==max_epochs:
            evaluation_start=time.perf_counter()
            metrics,detail=evaluate_model(model,data)
            cumulative_eval+=time.perf_counter()-evaluation_start
            row.update({'validation_ndcg10':metrics['ndcg@10'],'validation_recall10':metrics['recall@10'],
                        'scoring_seconds':metrics['scoring_seconds'],'metric_seconds':metrics['metric_seconds']})
            if metrics['ndcg@10'] > best+1e-5:
                best=metrics['ndcg@10']; best_epoch=epoch
                best_state=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
            print(f"{name} epoch {epoch:02d}: loss={row['data_loss']:.5f}, validation NDCG@10={metrics['ndcg@10']:.6f}, scoring={metrics['scoring_seconds']:.2f}s",flush=True)
        history.append(row)
        pd.DataFrame(history).to_csv(output/f'{name}_history.csv',index=False)
        if stale>=patience: break
    model.load_state_dict(best_state)
    metrics,detail=evaluate_model(model,data)
    detail.to_csv(output/f'{name}_users.csv',index=False)
    np.savez_compressed(output/f'{name}_weights.npz',**{k:v.detach().numpy() for k,v in best_state.items()})
    result={**config,**metrics,'run':name,'seed':seed,'selected_epoch':best_epoch,
            'epochs_run':epoch,'parameters':sum(p.numel() for p in model.parameters()),
            'training_seconds':cumulative_train,'validation_search_seconds':cumulative_eval,
            'total_seconds':time.perf_counter()-total_start,'weight_decay':1e-4,
            'batch_size':1024,'negatives_per_positive':1,'objective':model.objective}
    save_json(output/f'{name}_result.json',result)
    return result


def run_phase2(root,repeat_seeds=True,max_epochs=24,every=4):
    root=Path(root).resolve(); output=root/'outputs/phase2'; output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    before=time.perf_counter(); data=prepare(root)
    save_json(output/'content_audit.json',data['content_audit'])
    save_json(output/'tag_vocabulary.json',data['vocabulary'])
    np.savez_compressed(output/'content_features.npz',ids=data['feature_ids'],weights=data['feature_weights'])
    protocol={'grid':GRID,'tuning_seed':42,'repeat_seeds':[43,44] if repeat_seeds else [],
              'max_epochs':max_epochs,'validation_every':every,'patience_evaluations':3,
              'selection':'macro validation NDCG@10; minimum improvement 1e-5; ties choose lower learning rate',
              'optimizer':'Adam with coupled L2 weight_decay=1e-4','batch_size':1024,
              'negative_sampling':'one uniform train-warm artist absent from user TRAIN per positive per epoch; validation/test never excluded',
              'cold_score':'strictly below minimum warm model score; retained candidates, not random untrained ID scores',
              'device':'cpu','threads':4,'torch':torch.__version__,
              'windows_execution_verified':platform.system()=='Windows','test_file_read':False,
              'test_metrics_evaluated':False,'feature_build_seconds':time.perf_counter()-before}
    save_json(output/'protocol.json',protocol)
    results=[]; selected=[]
    for config in GRID:
        result=train_config(config,data,output,max_epochs=max_epochs,every=every)
        results.append(result)
        pd.DataFrame(results).to_csv(output/'all_runs.csv',index=False)
    tuning=pd.DataFrame(results)
    for family in tuning.family.unique():
        chosen=tuning[tuning.family==family].sort_values(['ndcg@10','lr'],ascending=[False,True]).iloc[0]
        config={k:chosen[k] for k in ('family','dim','lr')}; config['dim']=int(config['dim'])
        selected.append({'family':family,'dim':config['dim'],'lr':float(config['lr']),
                         'tuning_run':str(chosen['run']),'tuning_ndcg10':float(chosen['ndcg@10'])})
        if repeat_seeds:
            for seed in (43,44):
                results.append(train_config(config,data,output,seed,max_epochs=max_epochs,every=every))
                pd.DataFrame(results).to_csv(output/'all_runs.csv',index=False)
    save_json(output/'selected_configs.json',selected)
    save_json(output/'status.json',{'stage':'validation development complete','test_metrics_evaluated':False,
            'training_runs':len(results),'final_test_model_frozen':False})
    print(pd.DataFrame(results)[['family','lr','seed','selected_epoch','ndcg@10','recall@10','total_seconds']].to_string(index=False))
    return pd.DataFrame(results)
