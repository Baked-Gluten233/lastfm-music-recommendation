from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
    from musicrec.final_evaluation import run_final_once,fit_fixed_epochs
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Final-model tests require the phase2 CPU dependency')
class FinalProtocolTests(unittest.TestCase):
    def test_existing_evaluation_state_blocks_second_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'outputs/final').mkdir(parents=True)
            (root/'outputs/final/evaluation_state.json').write_text('{"status":"completed"}')
            with patch('musicrec.final_evaluation.verify_freeze',return_value=({},{})),patch('musicrec.final_evaluation.prepare_known_history',side_effect=AssertionError('must not read data')):
                with self.assertRaises(RuntimeError):run_final_once(root)

    def test_fixed_epochs_have_no_evaluation_callback(self):
        matrix=csr_matrix([[1,0,0,1,0,0],[0,1,0,1,0,0],[0,0,1,0,1,0]],dtype=float)
        data={'matrix':matrix,'users':np.array([0,0,1,1,2,2]),'items':np.array([0,3,1,3,2,4]),
              'warm':np.array([1,1,1,1,1,0],bool),'seen':matrix.toarray().astype(bool)}
        config={'family':'bpr_mf','dim':4,'lr':.01,'final_epochs':2}
        with tempfile.TemporaryDirectory() as directory,patch('musicrec.final_evaluation.evaluate',side_effect=AssertionError('final training cannot evaluate labels')):
            info=fit_fixed_epochs(config,data,Path(directory),seed=99)
            self.assertEqual(info['epochs'],2)
            self.assertTrue((Path(directory)/'bpr_mf_seed99_weights.npz').is_file())
            self.assertEqual(len((Path(directory)/'bpr_mf_seed99_refit_history.csv').read_text(encoding="utf-8").splitlines()),3)

if __name__=='__main__':unittest.main()
