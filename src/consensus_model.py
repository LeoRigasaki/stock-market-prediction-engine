import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone

from .config import Config


def load_model_scores(config: Optional[Config] = None) -> Dict[str, float]:
    """Load positive model ranking scores from the saved risk summary."""
    cfg = config or Config()
    risk_summary_path = cfg.PROCESSED_DATA_PATH / "day11_risk_summary.csv"
    if not risk_summary_path.exists():
        return {}

    try:
        risk_df = pd.read_csv(risk_summary_path)
        if risk_df.empty or 'Model' not in risk_df.columns or 'Sharpe_Ratio' not in risk_df.columns:
            return {}

        return {
            str(row['Model']): max(float(row['Sharpe_Ratio']), 0.05)
            for _, row in risk_df.iterrows()
        }
    except Exception as exc:
        logger.warning(f"Failed to load model scores: {exc}")
        return {}


class ConsensusMetaEnsemble:
    """Confidence-aware consensus ensemble built from existing model templates."""

    def __init__(
        self,
        base_models: Dict[str, Any],
        model_scores: Optional[Dict[str, float]] = None,
        top_k: int = 4,
        min_signal_pct: Optional[float] = None,
        dispersion_scale_pct: float = 0.75,
        recency_weight_start: float = 0.7,
        recency_weight_end: float = 1.3,
    ) -> None:
        self.config = Config()
        self.base_models = dict(base_models)
        self.model_scores = model_scores or {}
        self.top_k = top_k
        self.min_signal_pct = (
            self.config.signal_threshold_pct()
            if min_signal_pct is None else float(min_signal_pct)
        )
        self.dispersion_scale_pct = float(dispersion_scale_pct)
        self.recency_weight_start = float(recency_weight_start)
        self.recency_weight_end = float(recency_weight_end)
        self.fitted_models: Dict[str, Any] = {}
        self.active_model_names: List[str] = []

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            'base_models': self.base_models,
            'model_scores': self.model_scores,
            'top_k': self.top_k,
            'min_signal_pct': self.min_signal_pct,
            'dispersion_scale_pct': self.dispersion_scale_pct,
            'recency_weight_start': self.recency_weight_start,
            'recency_weight_end': self.recency_weight_end,
        }

    def set_params(self, **params: Any) -> "ConsensusMetaEnsemble":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _ranked_names(self) -> List[str]:
        ranked = sorted(
            self.base_models.keys(),
            key=lambda name: self.model_scores.get(name, 1.0),
            reverse=True,
        )
        return ranked[: self.top_k] if self.top_k > 0 else ranked

    def _safe_fit(self, model: Any, X: Any, y: Any, sample_weight: np.ndarray) -> Optional[Any]:
        try:
            fitted = clone(model)
        except Exception:
            try:
                fitted = copy.deepcopy(model)
            except Exception as exc:
                logger.warning(f"Unable to clone model template: {exc}")
                return None

        if hasattr(fitted, 'set_params'):
            try:
                fitted_params = fitted.get_params(deep=False)
                serial_params = {}
                if 'n_jobs' in fitted_params:
                    serial_params['n_jobs'] = 1
                if 'nthread' in fitted_params:
                    serial_params['nthread'] = 1
                if serial_params:
                    fitted.set_params(**serial_params)
            except Exception:
                pass

        try:
            fitted.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            try:
                fitted.fit(X, y)
            except Exception as exc:
                logger.warning(f"Model fit failed without sample weights: {exc}")
                return None
        except Exception as exc:
            logger.warning(f"Model fit failed: {exc}")
            return None

        return fitted

    def fit(self, X: Any, y: Any) -> "ConsensusMetaEnsemble":
        self.fitted_models = {}
        self.active_model_names = []
        sample_weight = np.linspace(self.recency_weight_start, self.recency_weight_end, len(X))

        for model_name in self._ranked_names():
            template = self.base_models[model_name]
            fitted = self._safe_fit(template, X, y, sample_weight)
            if fitted is None:
                continue
            self.fitted_models[model_name] = fitted
            self.active_model_names.append(model_name)

        return self

    def _prediction_models(self) -> Dict[str, Any]:
        if self.fitted_models:
            return self.fitted_models

        selected = {}
        for model_name in self._ranked_names():
            selected[model_name] = self.base_models[model_name]
        return selected

    def _combine_predictions(self, prediction_matrix: np.ndarray, model_names: List[str]) -> np.ndarray:
        if prediction_matrix.size == 0:
            return np.array([], dtype=float)

        raw_scores = np.array([self.model_scores.get(name, 1.0) for name in model_names], dtype=float)
        weights = raw_scores / raw_scores.sum() if raw_scores.sum() > 0 else np.ones(len(model_names)) / len(model_names)

        weighted_prediction = np.average(prediction_matrix, axis=1, weights=weights)
        prediction_dispersion = np.std(prediction_matrix, axis=1) if prediction_matrix.shape[1] > 1 else np.zeros(len(weighted_prediction))
        agreement = np.clip(1.0 - (prediction_dispersion / max(self.dispersion_scale_pct, 1e-6)), 0.0, 1.0)
        final_prediction = weighted_prediction * agreement
        final_prediction[np.abs(final_prediction) < self.min_signal_pct] = 0.0
        return final_prediction

    def predict(self, X: Any) -> np.ndarray:
        models = self._prediction_models()
        model_names: List[str] = []
        predictions: List[np.ndarray] = []

        for model_name, model in models.items():
            try:
                pred = np.asarray(model.predict(X), dtype=float)
                if pred.ndim == 0:
                    pred = pred.reshape(1)
                if np.any(~np.isfinite(pred)):
                    continue
                predictions.append(pred)
                model_names.append(model_name)
            except Exception as exc:
                logger.warning(f"Consensus component {model_name} prediction failed: {exc}")

        if not predictions:
            return np.zeros(len(X), dtype=float)

        prediction_matrix = np.column_stack(predictions)
        return self._combine_predictions(prediction_matrix, model_names)
