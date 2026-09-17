import torch
import unittest
import warnings
import run_f3_v2 as wrapper


class GroupedBinsTest(unittest.TestCase):
    def test_grouped_bins_bitwise_identical(self):
        generator=torch.Generator().manual_seed(20260916)
        numbers=torch.randn(10003,69,generator=generator)
        numbers[:,2]=0
        numbers[:,17]=torch.arange(len(numbers))%2
        numbers[:,30]=torch.round(numbers[:,30])
        numbers[:,62:]=(numbers[:,62:]>0).float()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            old=wrapper.frozen.adapter.fit_bins(numbers,62)
            grouped=wrapper.grouped_fit_bins(numbers,62)
        self.assertTrue(torch.equal(old[1],grouped[1]))
        self.assertTrue(torch.equal(old[2],grouped[2]))
        self.assertEqual(len(old[0]),len(grouped[0]))
        for left,right in zip(old[0],grouped[0]):
            self.assertTrue(torch.equal(left,right))


if __name__=='__main__':
    unittest.main()
