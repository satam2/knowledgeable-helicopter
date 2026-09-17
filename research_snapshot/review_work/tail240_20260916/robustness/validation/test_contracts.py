import unittest
import pandas as pd
from contracts import ID, FLIGHT, TIME, validate_stages, validate_declared_access, validate_prior_support, validate_exposure, reconstruct_declared_cohorts, validate_refit, validate_refit_access


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame([(1,None,'2025-01-01'),(2,None,'2025-02-01'),
                                  (3,'c','2025-03-01'),(4,'e','2025-04-01')],columns=[ID,FLIGHT,TIME])
        self.stages = dict(train=[1],stop=[2],calibration=[3],evaluation=[4])
        self.events = [dict(operation=op,label_or_fit_ids=[i]) for op,i in
                       [('gradient',1),('preprocessing',1),('stopping',2),('calibration',3)]]
        self.events += [dict(operation='freeze',state_sha256='state',prediction_sha256='prediction'),
                        dict(operation='evaluation',label_or_fit_ids=[4])]

    def test_clean_separation_and_null_flights(self):
        self.assertEqual(len(validate_stages(self.meta,self.stages)),4)
        self.assertTrue(validate_declared_access(self.events,self.stages))

    def test_missing_stopping_fails(self):
        del self.stages['stop']
        with self.assertRaisesRegex(ValueError,'stopping'):validate_stages(self.meta,self.stages)

    def test_related_flight_leak_fails(self):
        self.meta.loc[0,FLIGHT]='e'
        with self.assertRaisesRegex(ValueError,'Related flight'):validate_stages(self.meta,self.stages)

    def test_nonforward_and_duplicate_ids_fail(self):
        self.stages['train']=[4]
        with self.assertRaises(ValueError):validate_stages(self.meta,self.stages)
        self.stages['train']=[1,1]
        with self.assertRaises(ValueError):validate_stages(self.meta,self.stages)

    def test_hidden_columns_fail(self):
        with self.assertRaisesRegex(ValueError,'Only identifier'):
            validate_stages(self.meta.assign(TAXITIME_SEC_mvt=0),self.stages)

    def test_calibration_used_for_stopping_fails(self):
        self.events[2]['label_or_fit_ids']=[3]
        with self.assertRaisesRegex(ValueError,'outside its stage'):validate_declared_access(self.events,self.stages)

    def test_eval_preprocessing_fails(self):
        self.events[1]['label_or_fit_ids']=[4]
        with self.assertRaisesRegex(ValueError,'outside its stage'):validate_declared_access(self.events,self.stages)

    def test_eval_before_freeze_fails(self):
        self.events[-2:]=self.events[-2:][::-1]
        with self.assertRaisesRegex(ValueError,'before freeze'):validate_declared_access(self.events,self.stages)

    def test_post_score_selection_fails(self):
        self.events.append(dict(operation='stopping',label_or_fit_ids=[2]))
        with self.assertRaisesRegex(ValueError,'after freeze'):validate_declared_access(self.events,self.stages)

    def test_prior_earlier_month_passes(self):
        self.assertTrue(validate_prior_support(self.meta.iloc[:2],self.meta.iloc[2:3]))

    def test_same_month_prior_fails(self):
        self.meta.loc[1,TIME]='2025-03-01'
        with self.assertRaisesRegex(ValueError,'query-month'):validate_prior_support(self.meta.iloc[:2],self.meta.iloc[2:3])

    def test_whole_month_flight_and_outer_purge_fail(self):
        with self.assertRaisesRegex(ValueError,'related query'):
            validate_prior_support(self.meta.iloc[2:3],self.meta.iloc[3:],excluded_flight_ids=['c'])
        self.meta.loc[0,FLIGHT]='c'
        with self.assertRaisesRegex(ValueError,'related query'):validate_prior_support(self.meta.iloc[:2],self.meta.iloc[2:3])

    def test_historical_exposure_cannot_be_renamed(self):
        declaration=dict(historically_exposed=True,fresh_holdout=False,
                         reuse_previously_stopped_models=False,ranking_targets_accessed=False)
        self.assertTrue(validate_exposure(declaration))
        for key in declaration:
            changed={**declaration,key:not declaration[key]}
            with self.subTest(key=key),self.assertRaises(ValueError):validate_exposure(changed)

    def test_panel_purges_include_later_evaluation_months(self):
        rows=[(1,None,'2025-01-01'),(2,'late','2025-02-01'),(3,None,'2025-04-01'),
              (4,'late','2025-04-02'),(5,None,'2025-05-01'),(6,'late','2025-05-02'),
              (7,None,'2025-06-01'),(8,None,'2025-07-01'),(9,'late','2025-12-01')]
        meta=pd.DataFrame(rows,columns=[ID,FLIGHT,TIME])
        ids,purges=reconstruct_declared_cohorts(meta,'F1')
        self.assertEqual(ids,dict(train=[1],stop=[3],calibration=[5],evaluation=[7,8,9],refit=[1,3]))
        self.assertEqual(purges,dict(train=1,stop=1,calibration=1))
        ids['refit'].append(5)
        with self.assertRaisesRegex(ValueError,'source-ordered union'):validate_refit(meta,ids)

    def test_refit_fresh_preprocessing_and_stage_order(self):
        stages={**self.stages,'refit':[1,2]}
        events=[dict(operation=op,label_or_fit_ids=ids) for op,ids in
                [('gradient',[1]),('preprocessing',[1]),('stopping',[2]),
                 ('refit_preprocessing',[1,2]),('refit_gradient',[1,2]),('calibration',[3])]]
        events[1]['fit_cohort_sha256']='train'
        events[3]['fit_cohort_sha256']='refit'
        events+=self.events[-2:]
        self.assertTrue(validate_refit_access(events,stages))
        events[3]['fit_cohort_sha256']='train'
        with self.assertRaisesRegex(ValueError,'freshly fitted'):validate_refit_access(events,stages)
        events[3]['fit_cohort_sha256']='refit'
        events[2],events[3]=events[3],events[2]
        with self.assertRaisesRegex(ValueError,'before stopping'):validate_refit_access(events,stages)


if __name__ == '__main__':unittest.main()
