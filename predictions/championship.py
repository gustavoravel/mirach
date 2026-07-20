"""Empirical model championship with walk-forward evaluation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .algorithms import BaseForecaster, create_forecaster

logger = logging.getLogger(__name__)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    den = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(den > eps, np.abs(y_pred - y_true) / den, 0.0) * 100.0))


class ModelChampionship:
    """
    Rank forecasting algorithms via rolling-origin (walk-forward) backtest.
    Always includes naive baselines. Winner = lowest RMSE.
    """

    BASELINES = ('naive', 'seasonal_naive')

    def eligible_candidates(self, n: int, has_exog: bool = False) -> List[str]:
        """Light eligibility gates by series length (not ranking)."""
        cands: List[str] = ['naive', 'seasonal_naive']
        if n >= 20:
            cands.extend(['linear_regression', 'ridge', 'lasso', 'arima'])
        if n >= 30:
            cands.append('ets')
        if n >= 40:
            cands.extend(['random_forest', 'xgboost'])
            try:
                from .algorithms import LIGHTGBM_AVAILABLE
                if LIGHTGBM_AVAILABLE:
                    cands.append('lightgbm')
            except Exception:
                pass
        if n >= 60:
            try:
                from .algorithms import PROPHET_AVAILABLE
                if PROPHET_AVAILABLE:
                    cands.append('prophet')
            except Exception:
                pass
        # Deduplicate preserving order
        return list(dict.fromkeys(cands))

    def _ctor_kwargs(self, algorithm: str, series: pd.Series) -> Dict[str, Any]:
        if algorithm == 'arima':
            return {'use_auto_arima': True, 'order': None}
        if algorithm == 'seasonal_naive':
            season = 7
            if hasattr(series.index, 'inferred_freq') and series.index.inferred_freq:
                freq = str(series.index.inferred_freq).upper()
                if freq.startswith('M'):
                    season = 12
                elif freq.startswith('W'):
                    season = 52
            elif len(series) >= 24:
                season = 12
            return {'season_length': min(season, max(1, len(series) // 2))}
        if algorithm == 'ets':
            periods = 12 if len(series) >= 24 else max(2, min(7, len(series) // 3))
            return {'seasonal_periods': periods}
        if algorithm in ('xgboost', 'lightgbm', 'random_forest'):
            return {'n_estimators': 50, 'random_state': 42}
        return {}

    def walk_forward(
        self,
        series: pd.Series,
        algorithm: str,
        *,
        exog: Optional[pd.DataFrame] = None,
        n_windows: int = 3,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Rolling-origin evaluation for one algorithm."""
        n = len(series)
        if horizon is None:
            horizon = max(1, min(7, n // 10 or 1))
        min_train = max(10, n // 2)
        if n < min_train + horizon:
            # Single holdout split
            split = max(min_train, n - horizon)
            windows = [(0, split, split, min(n, split + horizon))]
        else:
            step = max(1, (n - min_train - horizon) // max(1, n_windows - 1))
            windows = []
            for w in range(n_windows):
                train_end = min_train + w * step
                test_end = min(n, train_end + horizon)
                if train_end >= n or test_end <= train_end:
                    break
                windows.append((0, train_end, train_end, test_end))

        maes, rmses, smapes = [], [], []
        last_error = None
        kwargs = self._ctor_kwargs(algorithm, series)
        # ARIMA with order=None triggers auto; remove explicit None issues
        if algorithm == 'arima' and kwargs.get('order') is None:
            kwargs.pop('order', None)
            # Force auto via constructor: order=None
            pass

        for start, train_end, test_start, test_end in windows:
            train = series.iloc[start:train_end]
            test = series.iloc[test_start:test_end]
            if len(train) < 5 or len(test) < 1:
                continue
            train_exog = exog.iloc[start:train_end] if exog is not None else None
            test_exog = exog.iloc[test_start:test_end] if exog is not None else None
            try:
                if algorithm == 'arima':
                    model = create_forecaster('arima', use_auto_arima=True)
                else:
                    model = create_forecaster(algorithm, **kwargs)
                fit_kw: Dict[str, Any] = {}
                if algorithm not in ('ets', 'naive', 'seasonal_naive') and train_exog is not None:
                    fit_kw['exog'] = train_exog
                model.fit(train, **fit_kw)
                pred_kw: Dict[str, Any] = {}
                if algorithm not in ('ets', 'naive', 'seasonal_naive') and test_exog is not None:
                    pred_kw['exog'] = test_exog
                preds = model.predict(len(test), **pred_kw)
                y_true = np.asarray(test.values, dtype=float)
                y_pred = np.asarray(preds.values[: len(test)], dtype=float)
                mae = float(np.mean(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                sm = _smape(y_true, y_pred)
                maes.append(mae)
                rmses.append(rmse)
                smapes.append(sm)
            except Exception as exc:
                last_error = str(exc)
                logger.debug("Championship window failed for %s: %s", algorithm, exc)
                continue

        if not rmses:
            return {
                'algorithm': algorithm,
                'rmse': float('inf'),
                'mae': float('inf'),
                'smape': float('inf'),
                'windows': 0,
                'error': last_error or 'no successful windows',
            }
        return {
            'algorithm': algorithm,
            'rmse': float(np.mean(rmses)),
            'mae': float(np.mean(maes)),
            'smape': float(np.mean(smapes)),
            'windows': len(rmses),
            'error': None,
        }

    def run(
        self,
        series: pd.Series,
        *,
        exog: Optional[pd.DataFrame] = None,
        candidates: Optional[List[str]] = None,
        n_windows: int = 3,
    ) -> Dict[str, Any]:
        series = pd.to_numeric(series, errors='coerce').dropna()
        if series.name is None:
            series = series.rename('target')
        n = len(series)
        has_exog = exog is not None and len(exog) > 0
        cands = candidates or self.eligible_candidates(n, has_exog=has_exog)
        # Always ensure baselines
        for b in self.BASELINES:
            if b not in cands:
                cands.insert(0, b)

        results = []
        for algo in cands:
            results.append(self.walk_forward(series, algo, exog=exog, n_windows=n_windows))

        ranking = sorted(results, key=lambda r: (r['rmse'], r['mae']))
        naive_rmse = next(
            (r['rmse'] for r in ranking if r['algorithm'] == 'naive'),
            float('inf'),
        )
        best = ranking[0] if ranking else None
        beats_baseline = bool(
            best
            and best['algorithm'] not in self.BASELINES
            and best['rmse'] < naive_rmse
            and best['rmse'] < float('inf')
        )
        # Prefer best non-baseline for recommendation when it wins; else still report best
        recommended = best
        if best and best['algorithm'] in self.BASELINES:
            non_base = [r for r in ranking if r['algorithm'] not in self.BASELINES and r['rmse'] < float('inf')]
            if non_base:
                recommended = non_base[0]

        return {
            'success': True,
            'ranking': ranking,
            'best_model': recommended['algorithm'] if recommended else None,
            'beats_baseline': beats_baseline,
            'naive_rmse': naive_rmse if naive_rmse < float('inf') else None,
            'n_points': n,
            'has_exog': has_exog,
        }
