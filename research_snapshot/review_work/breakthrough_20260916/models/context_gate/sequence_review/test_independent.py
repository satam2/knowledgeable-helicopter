"""Independent small sequence-model tests; no cache load or GPU work."""
import torch
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SEQUENCE = ROOT/'review_work/breakthrough_20260916/sequence_context'
sys.path.insert(0,str(SEQUENCE))
import adapter
import cache


class IndependentSequence(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(445)

    def test_padding_values_cannot_affect_valid_hidden_state(self):
        net = adapter.Network(4,[5],[5]*6+[2]*3).eval()
        qn,qc = torch.randn(2,4),torch.ones(2,1,dtype=torch.long)
        en,ec = torch.randn(2,32,22),torch.ones(2,32,9,dtype=torch.long)
        mask = torch.zeros(2,32,dtype=torch.bool)
        mask[0,:3] = True
        support = torch.tensor([[3/32,3/16,0.],[0.,0.,0.]])
        original = net(qn,qc,en,ec,mask,support,True)
        en[~mask] = 1000.
        ec[~mask] = 0
        changed = net(qn,qc,en,ec,mask,support,True)
        torch.testing.assert_close(original,changed,rtol=0.,atol=0.)
        static = net(qn,qc,en,ec,mask,support,False)
        torch.testing.assert_close(static[1],changed[1],rtol=0.,atol=0.)

    def test_unreferenced_event_mutation_cannot_change_fitted_encoder(self):
        n=4
        events=pd.DataFrame({'time_ns':np.arange(n)*10**9,'movement_ns':np.arange(n)*10**9,
                             'phase':['DEP']*n,'runway':['A']*n,'stand':['S']*n,'operator':['O']*n,
                             'aircraft':['A320']*n,'wake':['M']*n})
        for name in cache.NUMS:
            events[name]=np.arange(n,dtype=float)
        queries=pd.DataFrame({'time_ns':[5*10**9,6*10**9],'runway':['A','A'],'stand':['S','S'],'operator':['O','O']})
        neighbors=np.full((2,32),-1,dtype=np.int32)
        neighbors[0,:2]=[0,1]
        neighbors[1,:2]=[2,3]
        first=adapter.EventStore(events.copy(),queries,neighbors).fit_encoder(np.array([0]))
        events.loc[[2,3],cache.NUMS]=1e12
        events.loc[[2,3],'operator']='future_only'
        second=adapter.EventStore(events,queries,neighbors).fit_encoder(np.array([0]))
        np.testing.assert_array_equal(first['numeric'].mean,second['numeric'].mean)
        np.testing.assert_array_equal(first['numeric'].std,second['numeric'].std)
        self.assertEqual(first['cats'].maps,second['cats'].maps)
        self.assertNotIn('future_only',second['cats'].maps['operator'])

    def test_state_and_encoder_joblib_replay(self):
        network=adapter.Network(4,[5],[5]*6+[2]*3).eval()
        values=(torch.randn(2,4),torch.ones(2,1,dtype=torch.long),torch.randn(2,32,22),
                torch.ones(2,32,9,dtype=torch.long),torch.ones(2,32,dtype=torch.bool),torch.ones(2,3))
        original=network(*values,True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.joblib'
            joblib.dump(network.state_dict(),path)
            loaded=adapter.Network(4,[5],[5]*6+[2]*3).eval()
            loaded.load_state_dict(joblib.load(path))
        torch.testing.assert_close(original,loaded(*values,True),rtol=0.,atol=0.)


if __name__=='__main__':
    unittest.main(verbosity=2)
