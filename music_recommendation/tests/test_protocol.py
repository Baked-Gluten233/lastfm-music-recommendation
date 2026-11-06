"""Small exact tests: run python -m unittest discover -s tests -v."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musicrec.data import split_interactions
from musicrec.evaluation import evaluate, ranking_metrics
from musicrec.models import item_cosine, retain_neighbors, topk_unseen


class ProtocolTests(unittest.TestCase):
    def test_split_disjoint_complete_and_order_invariant(self):
        frame = pd.DataFrame([(u, i, i + 1) for u, n in enumerate([1, 4, 5, 9, 10, 50])
                              for i in range(n)], columns=["userID", "artistID", "weight"])
        one = split_interactions(frame)
        two = split_interactions(frame.sample(frac=1, random_state=7))
        for key in one:
            pd.testing.assert_frame_equal(one[key], two[key])
        pairs = [set(zip(d.userID, d.artistID)) for d in one.values()]
        self.assertEqual(len(set.union(*pairs)), len(frame))
        for a, b in ((0,1), (0,2), (1,2)):
            self.assertFalse(pairs[a] & pairs[b])
        self.assertEqual(one['train'].groupby('userID').size().tolist(), [1,4,3,7,8,40])
        self.assertEqual(one['validation'].groupby('userID').size().tolist(), [1,1,1,5])

    def test_cosine_shrinkage_integer_and_cold(self):
        train = csr_matrix([[1,1,0,0], [1,0,1,0], [1,1,0,0]])
        actual = item_cosine(train).toarray()
        self.assertAlmostEqual(actual[0,1], 2 / np.sqrt(6))
        self.assertAlmostEqual(actual[0,2], 1 / np.sqrt(3))
        self.assertTrue(np.all(actual.diagonal() == 0))
        self.assertTrue(np.all(actual[3] == 0))
        shrink = item_cosine(train, 2).toarray()
        self.assertAlmostEqual(shrink[0,1], actual[0,1] * 2/4)
        self.assertAlmostEqual(shrink[0,2], actual[0,2] * 1/3)

    def test_directed_neighbor_orientation(self):
        sim = csr_matrix([[0,.8,.7], [.8,0,.9], [.7,.9,0]])
        pruned = retain_neighbors(sim, 1)
        np.testing.assert_allclose((csr_matrix([[1,0,0]]) @ pruned).toarray(), [[0,.8,0]])
        self.assertEqual(pruned[1,0], 0)

    def test_exact_metrics(self):
        result = ranking_metrics([1,9,2], {1,2}, 3)
        self.assertAlmostEqual(result['precision'], 2/3)
        self.assertEqual(result['recall'], 1)
        self.assertAlmostEqual(result['ndcg'], .9197207891481876)
        self.assertEqual(result['hit_rate'], 1)

    def test_invalid_metric_inputs(self):
        for ranking, targets, k in [([1], {1}, 0), ([1,1], {1}, 2), ([1], set(), 1)]:
            with self.assertRaises(ValueError): ranking_metrics(ranking, targets, k)

    def test_seen_mask_and_ties(self):
        ranked = topk_unseen(np.zeros(5), np.array([0]), np.array([9,2,3,3,0]), 10)
        np.testing.assert_array_equal(ranked, [2,3,1,4])

    def test_cold_targets_retained(self):
        train = csr_matrix([[1,0,0], [0,1,0]], dtype=float)
        heldout = pd.DataFrame({'userID':[10,10], 'artistID':[101,102]})
        summary, detail = evaluate(train, heldout, {10:0,11:1}, {100:0,101:1,102:2},
                                   np.array([1.,1.,0.]), ks=(1,))
        self.assertEqual(summary['recall@1'], .5)
        self.assertEqual(summary['cold_targets'], 1)
        self.assertEqual(summary['catalogue_coverage@10'], 2/3)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(csr_matrix([[1.,0]]), pd.DataFrame({'userID':[1], 'artistID':[10]}),
                     {1:0}, {10:0,11:1}, np.array([1.,0.]))

    def test_neighbor_and_ks_validation(self):
        with self.assertRaises(ValueError): retain_neighbors(csr_matrix((2,2)), 0)
        with self.assertRaises(ValueError): item_cosine(csr_matrix((2,2)), -1)
        with self.assertRaises(ValueError): evaluate(csr_matrix((2,2)), None, {}, {}, np.zeros(2), ks=())

if __name__ == '__main__':
    unittest.main()
