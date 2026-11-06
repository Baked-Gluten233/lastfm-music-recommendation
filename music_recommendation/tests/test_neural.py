"""Leakage, shape, gradient, factorization, and negative-sampling tests."""
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from musicrec.content import build_tag_features
try:
    import torch
    from musicrec.neural import BPRMF,NeuMF,ContentTwoTower,sample_negatives,score_catalogue
except ImportError:
    torch=None


class ContentTests(unittest.TestCase):
    def fixture(self):
        train=pd.DataFrame({'userID':[1,1,2],'artistID':[10,11,12],'weight':[1,1,1]})
        events=pd.DataFrame({'userID':[1,1,2,1,7],'artistID':[10,11,12,12,999],
                             'tagID':[5,5,6,9999,7]})
        return train,events

    def test_heldout_pair_tags_have_no_effect(self):
        train,events=self.fixture()
        a=build_tag_features(events,train,[10,11,12,13],min_df=1,max_features=5,max_tags=3)
        poisoned=pd.concat([events,pd.DataFrame({'userID':[1,1],'artistID':[12,13],'tagID':[42,43]})])
        b=build_tag_features(poisoned,train,[10,11,12,13],min_df=1,max_features=5,max_tags=3)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(a[2],b[2]);self.assertNotIn(9999,a[2])
        self.assertEqual(a[3]['allowed_unique_train_tag_events'],3)
        self.assertEqual(a[3]['artists_without_content'],1)

    def test_vocab_threshold_and_unknown_vector(self):
        train,events=self.fixture()
        ids,weights,vocab,audit=build_tag_features(events,train,[10,11,12,13],min_df=2,max_tags=2)
        self.assertEqual(vocab,[5]);self.assertEqual(audit['artists_with_content'],2)
        np.testing.assert_array_equal(ids[2],[1,1]);np.testing.assert_array_equal(weights[2],[1.,0.])
        np.testing.assert_allclose(np.linalg.norm(weights,axis=1),np.ones(4))

    def test_no_content_gate(self):
        train,events=self.fixture()
        with self.assertRaises(ValueError):build_tag_features(events,train,[10,11,12,13],min_df=100)


@unittest.skipIf(torch is None,'Install optional PyTorch CPU dependency for phase2 tests')
class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11);torch.set_num_threads(1)

    def test_negative_sampler_only_train_information(self):
        seen=np.array([[True,False,False,False],[False,True,False,False]])
        users=np.tile([0,1],100)
        negative=sample_negatives(users,seen,np.array([0,1,2]),np.random.default_rng(7))
        self.assertFalse(seen[users,negative].any());self.assertNotIn(3,negative)
        # A potential validation positive remains eligible: no oracle exclusion.
        self.assertIn(1,negative[users==0])
        with self.assertRaises(ValueError):sample_negatives(np.array([0]),np.ones((1,2),bool),np.array([0,1]),np.random.default_rng(0))

    def test_bpr_formula_and_gradients(self):
        model=BPRMF(2,3,4);u=torch.tensor([0,1]);i=torch.tensor([1,2])
        torch.testing.assert_close(model(u,i),(model.user(u)*model.item(i)).sum(1))
        torch.nn.functional.softplus(-model(u,i)).mean().backward()
        self.assertGreater(float(model.user.weight.grad.abs().sum()),0)

    def test_neumf_separate_branches_train(self):
        model=NeuMF(2,3,4)
        self.assertIsNot(model.user_gmf,model.user_mlp)
        loss=torch.nn.functional.softplus(-model(torch.tensor([0,1]),torch.tensor([1,2]))).mean()
        loss.backward()
        self.assertGreater(float(model.user_gmf.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.user_mlp.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.fusion.weight.grad.abs().sum()),0)

    def test_content_towers_identical_content_and_factorization(self):
        ids=np.array([[0,2],[0,2],[2,2]]);weights=np.array([[1,0],[1,0],[1,0]],dtype=np.float32)
        model=ContentTwoTower(2,ids,weights,2,4)
        self.assertFalse(any(n.startswith('item.weight') for n,_ in model.named_parameters()))
        torch.testing.assert_close(model.encode_items(torch.tensor([0])),model.encode_items(torch.tensor([1])))
        torch.testing.assert_close(model(torch.tensor([0,1]),torch.tensor([1,2])),model.all_scores()[[0,1],[1,2]])
        (-model(torch.tensor([0]),torch.tensor([1]))).mean().backward()
        self.assertGreater(float(model.tag_embedding.weight.grad.abs().sum()),0)

    def test_cold_scores_and_batch_invariance(self):
        model=NeuMF(2,3,4);warm=np.array([True,True,False])
        a=score_catalogue(model,2,3,warm,pair_batch_size=2)
        b=score_catalogue(model,2,3,warm,pair_batch_size=6)
        np.testing.assert_allclose(a,b,atol=1e-6)
        self.assertTrue((a[:,2]<a[:,:2].min(axis=1)).all())

    def test_training_checkpoint_reproducibility(self):
        from scipy.sparse import csr_matrix
        from musicrec.phase2 import train_config
        matrix=csr_matrix([[1,0,0,1,0,0],[0,1,0,1,0,0],[0,0,1,0,1,0]],dtype=float)
        train_users=np.array([0,0,1,1,2,2]);train_items=np.array([0,3,1,3,2,4])
        data={'matrix':matrix,'users':train_users,'items':train_items,
              'warm':np.array([1,1,1,1,1,0],bool),'seen':matrix.toarray().astype(bool),
              'popularity':np.asarray(matrix.sum(0)).ravel(),
              'validation':pd.DataFrame({'userID':[10,11,12],'artistID':[101,102,100]}),
              'user_index':{10:0,11:1,12:2},'item_index':{100+i:i for i in range(6)}}
        config={'family':'bpr_mf','dim':4,'lr':.01}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            first=train_config(config,data,path,seed=123,max_epochs=2,every=1)
            with np.load(path/'bpr_mf_lr0.01_seed123_weights.npz') as stored:
                one={key:stored[key].copy() for key in stored.files}
            second=train_config(config,data,path,seed=123,max_epochs=2,every=1)
            with np.load(path/'bpr_mf_lr0.01_seed123_weights.npz') as stored:
                for key in stored.files:np.testing.assert_array_equal(one[key],stored[key])
            self.assertEqual(first['ndcg@10'],second['ndcg@10'])
            self.assertEqual(first['selected_epoch'],second['selected_epoch'])

if __name__=='__main__':unittest.main()
