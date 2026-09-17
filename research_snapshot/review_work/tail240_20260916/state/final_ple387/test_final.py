import unittest
import numpy as np


class FinalPolicyTests(unittest.TestCase):
    def test_existing_release_median_policy(self):
        self.assertEqual(int(np.median([20,19])),19)
        self.assertEqual(int(np.median([19,20])),19)

    def test_eight_event_request_chooses_extended_source(self):
        import sources
        import json
        from pathlib import Path
        marker=sources.common.read_json(sources.BASE/'missing/sequence_flatten8/manifest.json')
        requested=marker['features']
        tuples=sources.feature_sources(requested)
        self.assertEqual(len(tuples),2)
        self.assertTrue(all('flatten8' in str(p) or 'flat8' in str(p) for p,_,_,_ in tuples))
        self.assertTrue(all(not fill for _,_,fill,_ in tuples))
        self.assertTrue(all(cols==requested for _,_,_,cols in tuples))


if __name__=='__main__':unittest.main()
