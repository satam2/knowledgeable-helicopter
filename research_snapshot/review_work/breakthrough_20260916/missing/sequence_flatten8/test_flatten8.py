import unittest
import numpy as np
import pandas as pd
import build


class Extension(unittest.TestCase):
    def test_nearest_eight_and_original_prefix(self):
        events=pd.DataFrame({'time_ns':np.arange(24)*10**9,'movement_ns':np.arange(24)*10**9-10**9,
            'phase':['DEP','ARR']*12,'runway':'01','stand':'<missing>','operator':'m:'})
        for j,col in enumerate(build.cache.NUMS):
            events[col]=np.arange(24,dtype='float32')+j
        query=pd.DataFrame({'time_ns':[100*10**9],'runway':['01'],'stand':['<missing>'],'operator':['m:']})
        idx=np.array([list(range(24))+[-1]*8])
        engine=build.Flatten8(events,query)
        expected=[[22,20,18,16,14,12,10,8,23,21,19,17,15,13,11,9]]
        np.testing.assert_array_equal(engine.indices(idx),expected)
        full=pd.DataFrame(engine.batch(idx,0),columns=build.COLUMNS)
        prior=build.old.Flatten(events,query).batch(idx,0)
        np.testing.assert_allclose(full[build.old.COLUMNS],prior,rtol=0,atol=0,equal_nan=True)
        assert (full[[c for c in full if c.endswith(('same_stand','same_operator'))]]==0).all().all()

    def test_zero_and_sparse_context_padding(self):
        events=pd.DataFrame({'time_ns':[0],'movement_ns':[0],'phase':['DEP'],'runway':['X'],'stand':['Y'],'operator':['Z']})
        for col in build.cache.NUMS:
            events[col]=np.nan
        queries=pd.DataFrame({'time_ns':[10**9,10**9],'runway':['A','A'],'stand':['B','B'],'operator':['C','C']})
        idx=np.full((2,32),-1)
        idx[1,0]=0
        values=build.Flatten8(events,queries).batch(idx,0).reshape(2,16,14)
        assert np.isnan(values[0,:,:10]).all()
        np.testing.assert_array_equal(values[0,:,10:13],0)
        np.testing.assert_array_equal(values[0,:,13],1)
        self.assertEqual(values[1,0,13],0)
        np.testing.assert_array_equal(values[1,1:,13],1)


if __name__=='__main__':
    unittest.main()
