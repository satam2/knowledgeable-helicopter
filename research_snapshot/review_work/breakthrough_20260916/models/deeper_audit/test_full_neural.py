"""Bounded wrapper dispatch checks with synthetic data and no fit/GPU calls."""
import torch
import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PATH = ROOT / 'review_work/breakthrough_20260916/full_neural/run.py'
spec = importlib.util.spec_from_file_location('full_neural_review', PATH)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
core = wrapper.core


def feature_calls(path):
    result = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('augment', 'additional'):
            result.append((node.func.attr, ast.literal_eval(node.args[1])))
    return sorted(result)


class WrapperContracts(unittest.TestCase):
    def test_exact_matched_feature_block_order(self):
        self.assertEqual(feature_calls(PATH), feature_calls(ROOT / 'review_work/breakthrough_20260916/deeper_lgb/run.py'))

    def test_three_dispatches_preserve_finite_domain(self):
        marker = core.read_json(ROOT / 'private_runs/breakthrough_20260916/deeper_lgb/combined/lightgbm_leaf63_aobt_allfinite_F1_s20260916/manifest.json')
        columns = marker['feature_columns']
        self.assertEqual(len(columns), 225)
        frame = pd.DataFrame(np.zeros((4, len(columns))), columns=columns)
        meta = pd.DataFrame({'proxy_sec': [-12., 131167., np.nan, 500.]})
        for encoding in ['standard', 'ple8', 'ple32']:
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                observed = []

                def declare(args):
                    name = f'{args.family}_{args.formulation}_s{args.seed}'
                    (destination / 'source_snapshots' / name).mkdir(parents=True)
                    protocol = destination / 'protocol.json'
                    core.write_json(protocol, {'synthetic_review_only': True})
                    return protocol

                def run(args, fold, x, supplied_meta, offset, eligible, info, protocol):
                    observed.append(args.family)
                    self.assertEqual(args.formulation, 'aobt_allfinite')
                    self.assertEqual(list(x), columns)
                    self.assertEqual(fold, 'F1')
                    np.testing.assert_array_equal(eligible, [True, True, False, True])
                    np.testing.assert_array_equal(offset[[0, 1, 3]], [-12., 131167., 500.])
                    expected = wrapper.tabm_gpu if encoding == 'standard' else wrapper.tabm_ple_gpu
                    self.assertIs(core.importlib.import_module('tabm_gpu'), expected)
                    if encoding != 'standard':
                        self.assertEqual(expected.MEMBERS, int(encoding[3:]))
                    return {}

                with patch.object(core.common, 'load_data', return_value=(frame, meta)), \
                     patch.object(wrapper.run_augmented, 'augment', side_effect=lambda x, blocks: (x, [])), \
                     patch.object(wrapper.run_information, 'additional', side_effect=lambda x, blocks: (x, [])), \
                     patch.object(core, 'external_path', return_value=destination), \
                     patch.object(core, 'declare', side_effect=declare), patch.object(core, 'run', side_effect=run), \
                     patch.object(core, 'source_hashes', return_value={}), patch.object(core, 'importlib'), \
                     patch.object(core, 'OUT', destination), patch.object(wrapper.tabm_ple_gpu, 'MEMBERS', 8), \
                     patch.object(sys, 'argv', [str(PATH), '--encoding', encoding, '--folds', 'F1']):
                    wrapper.main()
                self.assertEqual(observed, ['tabm_combined_' + encoding])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
