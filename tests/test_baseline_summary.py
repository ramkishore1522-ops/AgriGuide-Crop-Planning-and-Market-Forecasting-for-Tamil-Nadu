import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from tn_no_lag_model import summarize_baseline_results


class BaselineSummaryTests(unittest.TestCase):
    def test_summarize_baseline_results_aggregates_metric_means(self):
        metrics = [
            {'MAE': 1.0, 'RMSE': 2.0, 'MAPE': 3.0, 'R2': 0.5},
            {'MAE': 3.0, 'RMSE': 4.0, 'MAPE': 5.0, 'R2': 0.9},
        ]

        summary = summarize_baseline_results(metrics, 'LSTM')

        self.assertEqual(summary['Model'].iloc[0], 'LSTM')
        self.assertAlmostEqual(summary['MAE_mean'].iloc[0], 2.0)
        self.assertAlmostEqual(summary['RMSE_mean'].iloc[0], 3.0)
        self.assertAlmostEqual(summary['MAPE_mean'].iloc[0], 4.0)
        self.assertAlmostEqual(summary['R2_mean'].iloc[0], 0.7)


if __name__ == '__main__':
    unittest.main()
