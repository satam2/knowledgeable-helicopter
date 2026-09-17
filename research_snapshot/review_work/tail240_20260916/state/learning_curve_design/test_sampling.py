import unittest
import numpy as np
from sampling import selected_positions, selected_vocab


class SamplingTests(unittest.TestCase):
    def test_nested_and_input_order_independent(self):
        ids=np.arange(100.,132.)
        small=ids[selected_positions(ids,8)]
        large=ids[selected_positions(ids,16)]
        self.assertTrue(set(small)<=set(large))
        np.testing.assert_array_equal(np.sort(ids[::-1][selected_positions(ids[::-1],8)]),small)
        self.assertTrue(np.all(np.diff(selected_positions(ids,8))>0))

    def test_reject_ambiguous_ids(self):
        for ids in [[1,1],[1.5,2],[1,np.nan],[1,2**54]]:
            with self.assertRaises(ValueError):selected_positions(ids,1)

    def test_only_selected_fit_owns_vocabulary(self):
        codes=np.array([2,3,4,0,1,4,3],dtype=np.float32)
        vocab,mapping=selected_vocab(codes,np.array([0,3,4]),['a','b','c'])
        self.assertEqual(vocab,['a'])
        np.testing.assert_array_equal(mapping[codes.astype(int)],[2,1,1,0,1,1,1])
        np.testing.assert_array_equal(codes,[2,3,4,0,1,4,3])


if __name__=='__main__':unittest.main()
