import unittest
import numpy as np
import pandas as pd
import build


class FlattenContract(unittest.TestCase):
    def fixture(self):
        events=pd.DataFrame({'time_ns':np.arange(10,dtype='int64')*10**9,
            'movement_ns':np.arange(10,dtype='int64')*10**9-10**9,
            'phase':['DEP','ARR']*5,'runway':['01']*10,'stand':['<missing>']*10,'operator':['OP']*10})
        for i,c in enumerate(build.cache.NUMS):
            events[c]=np.arange(10,dtype='float32')+i
        queries=pd.DataFrame({'time_ns':[20*10**9],'runway':['01'],'stand':['<missing>'],'operator':['unseen']})
        return build.Flatten(events,queries)

    def test_latest_four_each_phase_nearest_first(self):
        engine=self.fixture()
        indices=np.array([list(range(10))+[-1]*22])
        np.testing.assert_array_equal(engine.indices(indices),[[8,6,4,2,9,7,5,3]])
        output=engine.batch(indices,0).reshape(1,8,14)[0]
        np.testing.assert_array_equal(output[:,0],[8,6,4,2,9,7,5,3])
        np.testing.assert_array_equal(output[:,8],[12,14,16,18,11,13,15,17])
        np.testing.assert_array_equal(output[:,10],1)
        np.testing.assert_array_equal(output[:,11:13],0)
        np.testing.assert_array_equal(output[:,13],0)
        self.assertTrue(np.isnan(output[:4,9]).all())

    def test_padding_and_missing_are_distinct(self):
        engine=self.fixture()
        engine.numeric[0,:]=np.nan
        indices=np.array([[0]+[-1]*31])
        output=engine.batch(indices,0).reshape(1,8,14)[0]
        self.assertTrue(np.isnan(output[0,:8]).all())
        self.assertEqual(output[0,13],0)
        self.assertTrue(np.isnan(output[1:,:10]).all())
        np.testing.assert_array_equal(output[1:,10:13],0)
        np.testing.assert_array_equal(output[1:,13],1)


if __name__=='__main__':
    unittest.main()
