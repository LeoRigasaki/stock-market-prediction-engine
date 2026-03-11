import unittest

import numpy as np

from src.consensus_model import ConsensusMetaEnsemble
from src.risk_management import RiskManagementFramework


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class RiskManagementSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.framework = RiskManagementFramework()
        feature_df = cls.framework.load_feature_data()
        cls.filtered = feature_df[feature_df['Ticker'].isin(['AAPL', 'NVDA', 'MSFT'])].copy()
        cls.features, _, _ = cls.framework.prepare_portfolio_data(cls.filtered)
        cls.predictions = cls.framework.generate_predictions(
            cls.framework.load_best_models(),
            cls.features,
            cls.filtered,
        )

    def test_markowitz_portfolio_succeeds(self):
        result = self.framework.portfolio_optimization_markowitz(self.predictions, target_return=0.12)
        self.assertTrue(result.get('success'))
        self.assertIn('weights', result)
        self.assertEqual(len(result['stocks']), len(result['weights']))

    def test_risk_parity_portfolio_succeeds(self):
        result = self.framework.risk_parity_portfolio(self.predictions)
        self.assertTrue(result.get('success'))
        self.assertIn('weights', result)
        self.assertEqual(len(result['stocks']), len(result['weights']))

    def test_horizon_aware_annualization(self):
        returns = np.array([0.01, -0.005, 0.02, 0.0, 0.015], dtype=float)
        ratios = self.framework.calculate_sharpe_sortino_ratios(returns)
        expected_periods = self.framework.config.TRADING_DAYS_PER_YEAR / self.framework.config.FORECAST_HORIZON_DAYS
        expected_annual_return = returns.mean() * expected_periods

        self.assertAlmostEqual(ratios['periods_per_year'], expected_periods)
        self.assertAlmostEqual(ratios['annual_return'], expected_annual_return)

    def test_benchmark_suite_contains_core_baselines(self):
        benchmarks = self.framework.build_benchmark_suite(self.predictions)

        self.assertIn('EqualWeightLongOnly', benchmarks)
        self.assertIn('TopQuartileMomentum5D', benchmarks)
        self.assertIn('CrossSectionMomentum5D', benchmarks)
        self.assertGreater(benchmarks['EqualWeightLongOnly']['observations'], 0)
        self.assertIsInstance(
            benchmarks['TopQuartileMomentum5D']['performance_ratios']['sharpe_ratio'],
            float,
        )

    def test_consensus_model_holds_when_signal_is_too_small(self):
        ensemble = ConsensusMetaEnsemble(
            {
                'A': ConstantModel(0.10),
                'B': ConstantModel(0.12),
            },
            model_scores={'A': 1.0, 'B': 1.0},
            min_signal_pct=0.20,
            dispersion_scale_pct=0.50,
        )
        prediction = ensemble.predict(np.zeros((3, 2)))
        self.assertTrue(np.allclose(prediction, 0.0))


if __name__ == "__main__":
    unittest.main()
