import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.streamlit_dashboard import prediction_dataframe, strategy_curve_from_validation


class ProjectArtifactTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_required_artifacts_exist(self):
        required = [
            self.ROOT / "data" / "processed" / "day10_validation_results.json",
            self.ROOT / "data" / "processed" / "day11_risk_summary.csv",
            self.ROOT / "data" / "processed" / "day11_benchmark_summary.csv",
            self.ROOT / "data" / "processed" / "feature_importance_analysis.csv",
            self.ROOT / "data" / "processed" / "target_stocks.txt",
        ]
        for artifact in required:
            self.assertTrue(artifact.exists(), f"Missing artifact: {artifact}")

    def test_validation_results_have_core_sections(self):
        payload = json.loads((self.ROOT / "data" / "processed" / "day10_validation_results.json").read_text())
        for key in ['walk_forward', 'out_of_sample', 'robustness', 'risk_metrics', 'stability']:
            self.assertIn(key, payload)

    def test_performance_summary_contains_models(self):
        df = pd.read_csv(self.ROOT / "data" / "processed" / "day11_risk_summary.csv")
        self.assertGreaterEqual(len(df), 3)
        self.assertIn('Sharpe_Ratio', df.columns)
        self.assertIn('Annual_Return', df.columns)

    def test_benchmark_summary_contains_baselines(self):
        df = pd.read_csv(self.ROOT / "data" / "processed" / "day11_benchmark_summary.csv")
        self.assertGreaterEqual(len(df), 3)
        self.assertIn('Benchmark', df.columns)
        self.assertIn('Sharpe_Ratio', df.columns)

    def test_validation_curve_is_non_empty_and_finite(self):
        payload = json.loads((self.ROOT / "data" / "processed" / "day10_validation_results.json").read_text())
        curve_df = strategy_curve_from_validation(payload)

        self.assertFalse(curve_df.empty)
        self.assertTrue(np.isfinite(curve_df['StrategyValue']).all())
        self.assertTrue(curve_df['Date'].is_monotonic_increasing)

    def test_prediction_dataframe_handles_empty_rows(self):
        prediction_df = prediction_dataframe([])
        self.assertTrue(prediction_df.empty)


if __name__ == "__main__":
    unittest.main()
