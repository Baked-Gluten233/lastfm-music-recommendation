"""Small transparent PyTorch CPU implementations; no pretrained code or weights."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BPRMF(nn.Module):
    """User/item latent factors, dot-product score, trained by pairwise BPR."""
    objective = 'bpr'
    def __init__(self, users, items, dim=32):
        super().__init__()
        self.user = nn.Embedding(users,dim)
        self.item = nn.Embedding(items,dim)
        nn.init.normal_(self.user.weight,std=.05)
        nn.init.normal_(self.item.weight,std=.05)
    def forward(self, users, items):
        return (self.user(users)*self.item(items)).sum(-1)
    def all_scores(self):
        return self.user.weight @ self.item.weight.T


class NeuMF(nn.Module):
    """Separate GMF and MLP embeddings; nonlinear interaction + learned fusion."""
    objective = 'bce'
    def __init__(self, users, items, dim=16):
        super().__init__()
        self.user_gmf = nn.Embedding(users,dim)
        self.item_gmf = nn.Embedding(items,dim)
        self.user_mlp = nn.Embedding(users,dim)
        self.item_mlp = nn.Embedding(items,dim)
        self.mlp = nn.Sequential(nn.Linear(2*dim,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.fusion = nn.Linear(dim+16,1)
        for layer in [self.user_gmf,self.item_gmf,self.user_mlp,self.item_mlp]:
            nn.init.normal_(layer.weight,std=.05)
    def forward(self, users, items):
        gmf = self.user_gmf(users)*self.item_gmf(items)
        mlp = self.mlp(torch.cat([self.user_mlp(users),self.item_mlp(items)],dim=-1))
        return self.fusion(torch.cat([gmf,mlp],dim=-1)).squeeze(-1)


class ContentTwoTower(nn.Module):
    """Independent user encoder and tag-feature encoder, then cosine retrieval.

    Weighted EmbeddingBag is a sparse linear transform of TF-IDF content,
    followed by nonlinear item encoding. There is deliberately NO artist-ID
    embedding, so this is structurally distinct from MF and NeuMF.
    """
    objective = 'bpr'
    def __init__(self, users, feature_ids, feature_weights, vocabulary_size, dim=32):
        super().__init__()
        self.user = nn.Embedding(users,dim)
        self.user_tower = nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,dim))
        self.tag_embedding = nn.Embedding(vocabulary_size+1,64)
        self.item_tower = nn.Sequential(nn.ReLU(),nn.Linear(64,dim))
        self.register_buffer('feature_ids',torch.from_numpy(feature_ids))
        self.register_buffer('feature_weights',torch.from_numpy(feature_weights))
        nn.init.normal_(self.user.weight,std=.05)
        nn.init.normal_(self.tag_embedding.weight,std=.05)
    def encode_users(self,users):
        return F.normalize(self.user_tower(self.user(users)),dim=-1)
    def encode_items(self,items):
        features = F.embedding_bag(self.feature_ids[items],self.tag_embedding.weight,
                    per_sample_weights=self.feature_weights[items],mode='sum')
        return F.normalize(self.item_tower(features),dim=-1)
    def forward(self,users,items):
        return 10*(self.encode_users(users)*self.encode_items(items)).sum(-1)
    def all_scores(self):
        users = self.encode_users(torch.arange(self.user.num_embeddings))
        items = self.encode_items(torch.arange(len(self.feature_ids)))
        return 10*(users @ items.T)


class CrossFittedContentTwoTower(ContentTwoTower):
    """Shared towers with a fixed content bank for each user's assigned fold.

    A one-bank instance is the matched global-feature control. Parameters have
    identical shapes in both policies. The original model and checkpoints keep
    their existing constructor and state-dict layout.
    """
    def __init__(self, users, feature_ids, feature_weights, vocabulary_size,
                 user_folds, dim=32):
        if feature_ids.ndim != 3 or feature_ids.shape != feature_weights.shape:
            raise ValueError('Feature banks must have matching fold/item/tag shapes')
        if user_folds.shape != (users,):
            raise ValueError('Each user needs exactly one feature-bank assignment')
        banks, items, tags = feature_ids.shape
        if banks < 1 or items < 1 or tags < 1:
            raise ValueError('Feature banks must be nonempty')
        if np.any(user_folds < 0) or np.any(user_folds >= banks):
            raise ValueError('User fold is outside the feature banks')
        super().__init__(users, feature_ids.reshape(banks*items, tags),
                         feature_weights.reshape(banks*items, tags), vocabulary_size, dim)
        self.register_buffer('user_folds', torch.from_numpy(user_folds))
        self.catalogue_size = items
        self.bank_count = banks

    def forward(self, users, items):
        bank_items = self.user_folds[users]*self.catalogue_size + items
        return super().forward(users, bank_items)

    def all_scores(self):
        users = self.encode_users(torch.arange(self.user.num_embeddings))
        scores = users.new_empty((len(users), self.catalogue_size))
        item_ids = torch.arange(self.catalogue_size)
        for fold in range(self.bank_count):
            selected = self.user_folds == fold
            items = self.encode_items(item_ids + fold*self.catalogue_size)
            scores[selected] = 10*(users[selected] @ items.T)
        return scores


def sample_negatives(users, train_seen, warm_items, rng):
    """Uniform train-warm artists not in each user's TRAIN history; no oracle."""
    if np.any(train_seen.sum(axis=1)[users] >= len(warm_items)):
        raise ValueError('A user has no allowable negative')
    negatives = rng.choice(warm_items,len(users))
    mask = train_seen[users,negatives]
    while mask.any():
        negatives[mask] = rng.choice(warm_items,int(mask.sum()))
        mask = train_seen[users,negatives]
    return negatives


@torch.no_grad()
def score_catalogue(model, users, items, warm, pair_batch_size=65536):
    model.eval()
    if hasattr(model,'all_scores'):
        scores = model.all_scores().numpy().copy()
    else:
        # NeuMF cannot precompute independent item vectors for its joint MLP.
        scores = np.empty((users,items),dtype=np.float32)
        for start in range(0,users*items,pair_batch_size):
            flat = torch.arange(start,min(start+pair_batch_size,users*items))
            scores.ravel()[start:start+len(flat)] = model(flat//items,flat%items).numpy()
    # Cold ID parameters are untrained; use explicit common floor rather than
    # arbitrary random initialization. They remain full-catalogue candidates.
    if (~warm).any():
        scores[:,~warm] = scores[:,warm].min(axis=1,keepdims=True)-1
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model scores')
    return scores
