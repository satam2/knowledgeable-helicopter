"""Independent brute-force check of temporal and same-flight exclusions."""
import unittest
import numpy as np
import pandas as pd
import normalized_transfer_support as subject


class PriorSupportTest(unittest.TestCase):
    def test_against_rowwise_oracle(self):
        rng=np.random.default_rng(914)
        n=401
        frame=pd.DataFrame({subject.ID:np.arange(n), 'airport':rng.choice(['A','B'],n), 'schedule_bucket':rng.choice(['normal','large'],n), 'schedule_day':rng.integers(-1,2,n), 'schedule_missing':False, 'stand_missing':rng.integers(0,2,n), 'flight_prefix':rng.choice(['AA','BB'],n), 'day':rng.integers(0,15,n), 'FLIGHT_ID_mvt':rng.integers(0,25,n).astype(float), 'missing':rng.random(n)<.3, 'ge7200':rng.random(n)<.2, 'ge86400':rng.random(n)<.05})
        frame.loc[frame.index%7==0,'FLIGHT_ID_mvt']=np.nan
        for keys in subject.KEY_SETS.values():
            actual=subject.prior_support(frame,keys).set_index(subject.ID)
            for _,q in frame.loc[frame.missing].iterrows():
                keep=~frame.missing & frame.day.lt(q.day)
                for key in keys:
                    keep &= frame[key].eq(q[key])
                if pd.notna(q.FLIGHT_ID_mvt):
                    keep &= frame.FLIGHT_ID_mvt.ne(q.FLIGHT_ID_mvt)
                prior=frame.loc[keep]
                expected=[len(prior),prior.day.nunique(),prior.ge7200.sum(),prior.ge86400.sum()]
                np.testing.assert_array_equal(actual.loc[q[subject.ID]].to_numpy(),expected)


if __name__=='__main__':
    unittest.main()
