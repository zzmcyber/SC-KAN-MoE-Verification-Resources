import math
import unittest
import numpy as np
from evaluation.evaluate_predictions import metrics,evaluate


class MetricTests(unittest.TestCase):
    def test_standard_definitions(self):
        result=metrics([.1,.2,.3],[.1,.2,.4])
        self.assertAlmostEqual(result['RMSE'],math.sqrt(.01/3))
        self.assertAlmostEqual(result['R2'],.5)
        self.assertAlmostEqual(result['Bias'],1/30)
        self.assertAlmostEqual(result['Pearson_r'],math.sqrt(27/28))
        expected=1-math.sqrt((math.sqrt(27/28)-1)**2+(math.sqrt(7/3)-1)**2+(7/6-1)**2)
        self.assertAlmostEqual(result['KGE'],expected)

    def test_constant_observations(self):
        for prediction in ([.1,.2,.3],[.2,.2,.2]):
            result=metrics([.2,.2,.2],prediction)
            for name in ('R2','Pearson_r','KGE'):self.assertIsNone(result[name])
            self.assertTrue(math.isfinite(result['RMSE']))

    def test_constant_prediction_zero_mean_and_singleton(self):
        result=metrics([.1,.2,.3],[.2,.2,.2])
        self.assertAlmostEqual(result['R2'],0.)
        self.assertIsNone(result['Pearson_r']);self.assertIsNone(result['KGE'])
        self.assertIsNone(metrics([-1,0,1],[-.8,.1,.9])['KGE'])
        result=metrics([.1],[.2]);self.assertAlmostEqual(result['Bias'],.1);self.assertIsNone(result['R2'])

    def test_invalid_pairs_rejected(self):
        for obs,pred in [([],[]),([1,2],[1]),([1,np.nan],[1,2]),([1,2],[1,np.inf])]:
            with self.assertRaises(ValueError):metrics(obs,pred)
        row={'sample_id':'SITE_001_S0001','site_id':'SITE_001','observed_sm':'.2','predicted_sm':'.3'}
        with self.assertRaises(ValueError):evaluate([row,row])
        for field in ('sample_id','site_id','observed_sm','predicted_sm'):
            bad=dict(row);bad[field]=''
            with self.assertRaises(ValueError):evaluate([bad])

    def test_per_site_and_no_clipping(self):
        rows=[{'sample_id':'a','site_id':'A','observed_sm':.2,'predicted_sm':1.2},
              {'sample_id':'b','site_id':'B','observed_sm':.3,'predicted_sm':-.2}]
        result=evaluate(rows,True)
        self.assertEqual(set(result['sites']),{'A','B'})
        self.assertAlmostEqual(result['sites']['A']['Bias'],1.)
        self.assertAlmostEqual(result['sites']['B']['Bias'],-.5)

    def test_mixed_split_rejected(self):
        rows=[{'sample_id':'a','site_id':'A','observed_sm':.2,'predicted_sm':.3,'split':'train'},
              {'sample_id':'b','site_id':'B','observed_sm':.3,'predicted_sm':.4,'split':'test'}]
        with self.assertRaises(ValueError):evaluate(rows)


if __name__=='__main__':unittest.main()
