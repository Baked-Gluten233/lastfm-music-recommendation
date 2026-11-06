"""Content features exclusively from training-authorized user-artist tag events."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags, save_npz

from .data import save_json


def build_tag_features(events, train, artist_ids, min_df=2, max_features=2000, max_tags=32):
    """Return padded sparse TF-IDF feature IDs/weights for a genuine content tower.

    Tag *text* is not needed; numerical tag IDs preserve identity without guessing
    file encoding. The function has no validation/test input. There is no item-ID
    embedding. Unknown content shares a single explicit unknown feature.
    """
    if min_df < 1 or max_features < 1 or max_tags < 1:
        raise ValueError('Feature limits must be positive')
    catalogue = set(map(int, artist_ids))
    within = events.loc[events.artistID.isin(catalogue), ['userID','artistID','tagID']]
    allowed = within.merge(train[['userID','artistID']], on=['userID','artistID'],
                           how='inner', validate='many_to_one')
    # Repeated applications by the same user do not inflate a tag count.
    allowed = allowed.drop_duplicates(['userID','artistID','tagID'])
    documents = allowed[['artistID','tagID']].drop_duplicates()
    df = documents.groupby('tagID').size().rename('df').reset_index()
    df = df[df.df >= min_df].sort_values(['df','tagID'], ascending=[False,True]).head(max_features)
    if df.empty:
        raise ValueError('No train-only tags survive; content model cannot proceed')
    vocabulary = df.tagID.astype(int).tolist()
    item_index = {int(a): i for i,a in enumerate(artist_ids)}
    tag_index = {tag:i for i,tag in enumerate(vocabulary)}
    kept = allowed[allowed.tagID.isin(vocabulary)]
    matrix = coo_matrix((np.ones(len(kept)),
              (kept.artistID.map(item_index),kept.tagID.map(tag_index))),
              shape=(len(artist_ids),len(vocabulary))).tocsr()
    # IDF corpus consists of training-observed artists, not held-out targets.
    n_docs = train.artistID.nunique()
    frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = np.log((1 + n_docs) / (1 + frequency)) + 1
    matrix.data = 1 + np.log(matrix.data)
    matrix = (matrix @ diags(idf)).tocsr()
    padded_ids = np.full((len(artist_ids),max_tags), len(vocabulary),dtype=np.int64)
    padded_weights = np.zeros((len(artist_ids),max_tags),dtype=np.float32)
    lengths = np.zeros(len(artist_ids),dtype=int)
    for row in range(len(artist_ids)):
        a,b = matrix.indptr[row:row+2]
        cols, vals = matrix.indices[a:b], matrix.data[a:b]
        order = np.lexsort((cols,-vals))[:max_tags]
        lengths[row] = len(order)
        if len(order):
            values = vals[order]
            values = values / np.linalg.norm(values)
            padded_ids[row,:len(order)] = cols[order]
            padded_weights[row,:len(order)] = values
        else:
            padded_weights[row,0] = 1.0
    audit = {
        'raw_tag_events':len(events), 'outside_catalogue_events':int((~events.artistID.isin(catalogue)).sum()),
        'within_catalogue_events':len(within), 'allowed_unique_train_tag_events':len(allowed),
        'kept_train_tag_events':len(kept), 'vocabulary_size':len(vocabulary),
        'artists_with_content':int((lengths > 0).sum()),
        'artists_without_content':int((lengths == 0).sum()),
        'training_artists':int(n_docs), 'min_df':min_df, 'max_features':max_features,'max_tags_per_artist':max_tags,
        'tag_identity':'numeric tagID; no text encoding assumptions used in model',
        'idf_documents':'training-observed artists only',
        'eligibility':'tag (userID,artistID) must occur in train.csv; no held-out tag pairs',
        'cold_start_claim':False,
    }
    return padded_ids,padded_weights,vocabulary,audit
