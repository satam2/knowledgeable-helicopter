"""Focused source exclusion, raw-label reconstruction and chronological contracts."""
import lightgbm
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_models as runner
from aviation.test_domain_integration import frame


class PhysicalTests(unittest.TestCase):
    def test_cache_verification_is_bound_to_actual_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/'features.parquet'
            pd.DataFrame({'x':[1.]}).to_parquet(data)
            runner.write_json(root/'manifest.json',dict(status='complete',outputs={'features.parquet':runner.sha256(data)}))
            runner.write_json(root/'verification.json',dict(status='passed',manifest_sha256='stale'))
            with self.assertRaisesRegex(ValueError,'stale'):
                runner.read_block(root,'features.parquet',verification=True)
            runner.write_json(root/'verification.json',dict(status='passed',manifest_sha256=runner.sha256(root/'manifest.json')))
            actual,_ = runner.read_block(root,'features.parquet',verification=True)
            self.assertEqual(actual.x.iloc[0],1.)
            pd.DataFrame({'x':[2.]}).to_parquet(data)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                runner.read_block(root,'features.parquet',verification=True)

    def test_arm_exclusions_and_source_flag(self):
        x = frame()
        for col in ['proxy_missing','batch_surface_T_nm_proxy_inventory','histgeom_mapped_graph_shortest_m','weather_T_wind_speed_kt']:
            x[col] = 0.
        base = runner.arm_columns(x,'basephysicalfull')
        self.assertIn('proxy_missing',base)
        self.assertFalse(any(c.startswith(('batch_','histgeom_','weather_')) for c in base))
        self.assertNotIn('weather_T_wind_speed_kt',runner.arm_columns(x,'surfacegeometry'))
        x['takeoff_minus_AOBT_3_flt'] = 0.
        with self.assertRaisesRegex(ValueError,'Forbidden'):
            runner.arm_columns(x,'weather')

    def test_alignment_rejects_permutation(self):
        x = frame()
        ext = pd.DataFrame({'new':0.},index=x.index[::-1])
        with self.assertRaisesRegex(ValueError,'order'):
            runner.append_block(x,ext,['new'])

    def test_raw_labels_preserved_and_future_prior_rejected(self):
        x = frame()
        y = np.r_[[-50.,90000.],np.full(len(x)-2,700.)]
        captured = {}
        original = runner.adapter.lgb.LGBMRegressor.fit
        def spy(model,features,labels,**kwargs):
            captured['labels'] = np.asarray(labels).copy()
            captured['features'] = features.copy()
            return original(model,features,labels,**kwargs)
        with patch.object(runner.adapter.lgb.LGBMRegressor,'fit',spy):
            model,info = runner.adapter.fit(x,y,steps=3,threads=1)
        np.testing.assert_allclose(captured['labels']+captured['features'].physical_reference_q10_sec,y)
        self.assertEqual(info['rows'],len(y))
        self.assertNotIn('__movement_ns',model['encoder'].columns)
        with self.assertRaisesRegex(ValueError,'overlap or precede'):
            runner.adapter.predict(model,x)
        later = frame(('03',),20,9000)
        pred = runner.adapter.predict(model,later)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.joblib'
            joblib.dump(model,path)
            np.testing.assert_array_equal(pred,runner.adapter.predict(joblib.load(path),later))
        order = np.arange(len(later))[::-1]
        np.testing.assert_array_equal(pred[order],runner.adapter.predict(model,later.iloc[order]))

    def test_tuning_and_fresh_refit(self):
        fit = frame(('01','02'),120)
        tune = frame(('03',),120,3000)
        model,info = runner.adapter.fit(fit,np.full(len(fit),700.),(tune,np.full(len(tune),710.)),threads=1)
        full = pd.concat([fit,tune])
        refit,_ = runner.adapter.fit(full,np.full(len(full),705.),steps=info['steps'],threads=1)
        self.assertEqual(refit['prior'].n,len(full))
        self.assertEqual(model['prior'].n,len(fit))
        self.assertGreater(refit['prior'].last_fit,model['prior'].last_fit)


if __name__ == '__main__':
    unittest.main()
