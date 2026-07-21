"""
Prediction services for Django integration
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from django.core.files.base import ContentFile
import json
import io
import base64
from datetime import datetime, timedelta
import os
import math

from .algorithms import create_forecaster, ModelEvaluator
from .models import Prediction, PredictionModel, PredictionResult
from datasets.models import Dataset, ColumnMapping


class PredictionService:
    """Service for handling prediction operations"""
    
    def __init__(self):
        self.available_models = {
            'arima': 'ARIMA',
            'ets': 'ETS (Exponential Smoothing)',
            'prophet': 'Prophet',
            'lstm': 'LSTM',
            'linear_regression': 'Regressão Linear',
            'ridge': 'Regressão Ridge',
            'lasso': 'Regressão Lasso',
            'random_forest': 'Random Forest',
            'svr': 'Support Vector Regression',
            'neural_network': 'Rede Neural',
            'xgboost': 'XGBoost'
        }
    
    def prepare_data(self, dataset: Dataset) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Prepare target series and optional exogenous feature frame."""
        try:
            # Read the dataset file (supports remote storage)
            ext = (dataset.file_type or '').lower() or os.path.splitext(dataset.file.name)[1].lower()
            with dataset.file.open('rb') as fh:
                if ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(fh)
                elif ext == '.csv':
                    df = pd.read_csv(fh, low_memory=False)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
            
            # Get column mappings
            mappings = dataset.column_mappings.all()
            
            # Find timestamp and target columns
            timestamp_col = None
            target_col = None
            feature_cols = []
            dayfirst = True
            if getattr(dataset, 'data_profile', None) and isinstance(dataset.data_profile, dict):
                dayfirst = dataset.data_profile.get('dayfirst', True)
            
            for mapping in mappings:
                if mapping.column_type == 'timestamp':
                    timestamp_col = mapping.column_name
                elif mapping.column_type == 'target':
                    target_col = mapping.column_name
                elif mapping.column_type == 'feature':
                    feature_cols.append(mapping.column_name)
            
            if not timestamp_col or not target_col:
                raise ValueError("Timestamp and target columns must be mapped")
            
            # Convert timestamp column (prefer BR dayfirst when indicated)
            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col], errors='coerce', dayfirst=bool(dayfirst)
            )
            df = df.dropna(subset=[timestamp_col])
            df = df.set_index(timestamp_col)
            
            # Sort by timestamp and drop duplicate index
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            
            # Get target series
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            target_series = df[target_col].dropna()
            if target_series.name is None:
                target_series = target_series.rename(target_col)
            
            if len(target_series) == 0:
                raise ValueError(f"Target column '{target_col}' has no valid numeric data after conversion")

            exog_df = None
            if feature_cols:
                present = [c for c in feature_cols if c in df.columns]
                if present:
                    exog_df = df.loc[target_series.index, present].apply(pd.to_numeric, errors='coerce')
                    exog_df = exog_df.ffill().bfill()
                    # Drop all-null columns
                    exog_df = exog_df.dropna(axis=1, how='all')
                    if exog_df.shape[1] == 0:
                        exog_df = None
            
            return target_series, exog_df
            
        except Exception as e:
            raise ValueError(f"Data preparation failed: {str(e)}")
    
    def split_data(self, data: pd.Series, train_size: float = 0.8, 
                   validation_size: float = 0.1,
                   exog: Optional[pd.DataFrame] = None
                   ) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Split data (and optional exog) into train, validation, and test sets"""
        n = len(data)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + validation_size))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        train_exog = val_exog = test_exog = None
        if exog is not None:
            train_exog = exog.iloc[:train_end]
            val_exog = exog.iloc[train_end:val_end]
            test_exog = exog.iloc[val_end:]
        
        return train_data, val_data, test_data, train_exog, val_exog, test_exog
    
    def run_prediction(self, prediction: Prediction, *, finalize: bool = True) -> Dict[str, Any]:
        """Run prediction using the specified model"""
        try:
            # Update status
            prediction.status = 'training'
            prediction.save()
            
            # Prepare data
            target_series, exog_df = self.prepare_data(prediction.dataset)
            
            # Split data
            train_data, val_data, test_data, train_exog, val_exog, test_exog = self.split_data(
                target_series, 
                prediction.train_size, 
                prediction.validation_size,
                exog=exog_df,
            )
            
            # Create forecaster
            params = dict(prediction.model_parameters or {})
            # frequency is not a model ctor arg
            frequency = params.pop('frequency', None)
            forecaster = create_forecaster(
                prediction.prediction_model.algorithm_type,
                **params
            )

            # Validate minimum samples for ML models (lags/rolling drop initial rows)
            ml_models = {'linear_regression','ridge','lasso','random_forest','svr','neural_network','xgboost','lightgbm'}
            if prediction.prediction_model.algorithm_type in ml_models:
                if len(train_data) < 15:
                    raise ValueError("Insufficient training samples for ML model. Provide at least ~15-20 points.")
            
            # Fit model (ETS ignores exog by design)
            algo = prediction.prediction_model.algorithm_type
            fit_kwargs = {}
            if algo != 'ets' and train_exog is not None:
                fit_kwargs['exog'] = train_exog
            fit_results = forecaster.fit(train_data, **fit_kwargs)
            
            # Make predictions
            pred_kwargs = {}
            if algo != 'ets' and exog_df is not None:
                # No future exog available — models hold last values
                pred_kwargs['exog'] = None
            predictions = forecaster.predict(prediction.prediction_horizon, **pred_kwargs)
            
            # Evaluate on validation set if available
            metrics = None
            if len(val_data) > 0:
                val_pred_kwargs = {}
                if algo != 'ets' and val_exog is not None:
                    val_pred_kwargs['exog'] = val_exog
                # Re-fit on train only already done; predict val horizon
                val_predictions = forecaster.predict(len(val_data), **val_pred_kwargs)
                metrics = forecaster.evaluate(val_data, val_predictions)
            # Honest metrics: omit (null) when no validation set — never fake zeros
            
            # Explainability from fit_results / forecaster attributes
            explanation = {}
            try:
                fi = getattr(forecaster, 'feature_importances_', None)
                if not fi and isinstance(fit_results, dict):
                    fi = fit_results.get('feature_importance')
                if fi:
                    # Keep top 15 for payload size
                    if isinstance(fi, dict):
                        ranked = sorted(fi.items(), key=lambda x: abs(float(x[1])), reverse=True)[:15]
                        explanation['feature_importances'] = {k: float(v) for k, v in ranked}
                if hasattr(forecaster, 'coef_'):
                    explanation['coefficients'] = getattr(forecaster, 'coef_', None)
            except Exception:
                pass

            # Sanitize metrics (replace NaN/Inf with None for JSON)
            metrics = self._sanitize_json(metrics) if metrics is not None else None

            # Prediction intervals
            lower = upper = None
            try:
                if hasattr(forecaster, 'get_prediction_intervals'):
                    import inspect as _insp
                    sig = _insp.signature(forecaster.get_prediction_intervals)
                    if 'predictions' in sig.parameters:
                        intervals = forecaster.get_prediction_intervals(predictions)
                    else:
                        intervals = forecaster.get_prediction_intervals()
                    if intervals is not None:
                        lower, upper = intervals
            except Exception:
                lower = upper = None

            # Create prediction results
            self._create_prediction_results(
                prediction, predictions, target_series, frequency or params.get('frequency'),
                lower=lower, upper=upper,
            )
            
            # Update prediction payload; finalize status only when requested
            # (Celery task delays finalize until NarrativeAgent finishes)
            if finalize:
                prediction.status = 'completed'
                prediction.completed_at = datetime.now()
                prediction.progress = 100
            else:
                prediction.status = 'training'
                prediction.progress = 85
            prediction.metrics = metrics or {}
            prediction.predictions_data = self._sanitize_json({
                'forecast': predictions.tolist(),
                'forecast_dates': self._generate_forecast_dates(
                    target_series.index[-1], 
                    prediction.prediction_horizon,
                    frequency
                ),
                'fit_results': fit_results,
                'lower': lower.tolist() if lower is not None else None,
                'upper': upper.tolist() if upper is not None else None,
            })
            # Versioning snapshot (lightweight info)
            prediction.model_version = prediction.prediction_model.algorithm_type
            prediction.dataset_version = str(len(target_series))
            if len(target_series) > 0:
                prediction.dataset_snapshot = {
                    'start': str(target_series.index[0]),
                    'end': str(target_series.index[-1]),
                    'count': len(target_series)
                }
            # Merge explainability — keep domain / auto_plan / championship from wizard
            prev_expl = prediction.explainability if isinstance(prediction.explainability, dict) else {}
            merged = dict(prev_expl)
            if isinstance(explanation, dict):
                merged.update(explanation)
            # Ensure domain is present from dataset if missing
            if not merged.get('domain_code'):
                try:
                    from predictions.domains import resolve_dataset_domain
                    dom = resolve_dataset_domain(prediction.dataset)
                    merged['domain_code'] = dom.get('code')
                    merged['domain_label'] = dom.get('label')
                except Exception:
                    pass
            if not finalize:
                merged['insights_status'] = 'pending'
            prediction.explainability = merged
            prediction.save()
            # Webhook notification
            try:
                project = prediction.project
                if getattr(project, 'webhook_url', None):
                    import requests
                    payload = {
                        'event': 'prediction.completed',
                        'prediction_id': prediction.id,
                        'project_id': project.id,
                        'status': prediction.status,
                        'metrics': metrics,
                        'completed_at': prediction.completed_at.isoformat(),
                    }
                    requests.post(project.webhook_url, json=payload, timeout=5)
            except Exception:
                pass
            
            return self._sanitize_json({
                'success': True,
                'predictions': predictions.tolist(),
                'metrics': metrics,
                'fit_results': fit_results
            })
            
        except Exception as e:
            prediction.status = 'failed'
            prediction.error_message = str(e)
            prediction.save()
            
            return {
                'success': False,
                'error': str(e)
            }

    def _sanitize_json(self, obj: Any) -> Any:
        """Recursively convert NaN/Inf to None and numpy types to native for JSONField."""
        try:
            if obj is None:
                return None
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, (int, str, bool)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [self._sanitize_json(x) for x in obj]
            if isinstance(obj, dict):
                return {k: self._sanitize_json(v) for k, v in obj.items()}
            # numpy types
            try:
                import numpy as np  # local import safe
                if isinstance(obj, (np.floating,)):
                    val = float(obj)
                    if math.isnan(val) or math.isinf(val):
                        return None
                    return val
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.ndarray,)):
                    return [self._sanitize_json(x) for x in obj.tolist()]
            except Exception:
                pass
            return obj
        except Exception:
            return obj
    
    def _create_prediction_results(self, prediction: Prediction, 
                                 forecasts: pd.Series, original_data: pd.Series,
                                 frequency: Optional[str] = None,
                                 lower: Optional[pd.Series] = None,
                                 upper: Optional[pd.Series] = None):
        """Create PredictionResult objects"""
        # Clear existing results
        prediction.results.all().delete()
        
        # Generate forecast dates
        last_date = original_data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            try:
                if frequency:
                    freq_map = {
                        'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q', 'Y': 'Y'
                    }
                    freq = freq_map.get(frequency, 'D')
                    forecast_dates = pd.date_range(start=last_date, periods=len(forecasts)+1, freq=freq)[1:]
                else:
                    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecasts))]
            except Exception:
                forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecasts))]
        else:
            forecast_dates = [last_date + i + 1 for i in range(len(forecasts))]
        
        # Create results
        for i, (date, forecast_value) in enumerate(zip(forecast_dates, forecasts)):
            lo = float(lower.iloc[i]) if lower is not None and i < len(lower) else None
            hi = float(upper.iloc[i]) if upper is not None and i < len(upper) else None
            PredictionResult.objects.create(
                prediction=prediction,
                timestamp=date,
                predicted_value=float(forecast_value),
                confidence_interval_lower=lo,
                confidence_interval_upper=hi,
            )
    
    def _generate_forecast_dates(self, last_date: pd.Timestamp, horizon: int, frequency: Optional[str] = None) -> List[str]:
        """Generate forecast dates"""
        if isinstance(last_date, pd.Timestamp):
            try:
                if frequency:
                    freq_map = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q', 'Y': 'Y'}
                    freq = freq_map.get(frequency, 'D')
                    dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
                else:
                    dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
            except Exception:
                dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        else:
            dates = [last_date + i + 1 for i in range(horizon)]
        
        return [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) for date in dates]
    
    def compare_models(self, dataset: Dataset, models: List[str], 
                      train_size: float = 0.8) -> Dict[str, Any]:
        """Compare multiple models on the same dataset"""
        try:
            # Prepare data
            target_series, exog_df = self.prepare_data(dataset)
            
            # Split data
            train_data, val_data, test_data, train_exog, val_exog, test_exog = self.split_data(
                target_series, train_size, exog=exog_df
            )
            
            # Create forecasters
            forecasters = []
            for model_type in models:
                try:
                    forecaster = create_forecaster(model_type)
                    forecasters.append(forecaster)
                except Exception as e:
                    print(f"Failed to create {model_type}: {e}")
                    continue
            
            # Compare models
            comparison_df = ModelEvaluator.compare_models(forecasters, train_data, test_data)
            
            # Get best model
            best_model = comparison_df.iloc[0]
            
            return {
                'success': True,
                'comparison': comparison_df.to_dict('records'),
                'best_model': best_model.to_dict(),
                'recommendation': best_model['Model']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_recommendations(self, dataset: Dataset) -> Dict[str, Any]:
        """Empirical recommendations via ModelChampionship (with characteristic flags)."""
        try:
            from .championship import ModelChampionship

            target_series, exog_df = self.prepare_data(dataset)
            data_length = len(target_series)
            has_trend = self._detect_trend(target_series)
            has_seasonality = self._detect_seasonality(target_series)
            is_stationary = self._is_stationary(target_series)

            champ = ModelChampionship()
            result = champ.run(target_series, exog=exog_df)

            recommendations = result.get('ranking', [])
            # JSON-safe ranking (no inf)
            safe_ranking = []
            for r in recommendations:
                safe_ranking.append({
                    'algorithm': r.get('algorithm'),
                    'rmse': None if r.get('rmse') in (float('inf'), None) or (isinstance(r.get('rmse'), float) and r['rmse'] == float('inf')) else r.get('rmse'),
                    'mae': None if r.get('mae') == float('inf') else r.get('mae'),
                    'smape': None if r.get('smape') == float('inf') else r.get('smape'),
                    'windows': r.get('windows'),
                    'error': r.get('error'),
                })
            return {
                'success': True,
                'recommendations': [r['algorithm'] for r in safe_ranking if r.get('rmse') is not None][:5] or [r['algorithm'] for r in safe_ranking[:5]],
                'ranking': safe_ranking,
                'best_model': result.get('best_model'),
                'beats_baseline': result.get('beats_baseline'),
                'data_characteristics': {
                    'length': data_length,
                    'has_trend': has_trend,
                    'has_seasonality': has_seasonality,
                    'is_stationary': is_stationary,
                    'has_exog': exog_df is not None,
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _detect_trend(self, data: pd.Series) -> bool:
        """Detect if data has a trend"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            period = min(12, max(2, len(data) // 2))
            decomposition = seasonal_decompose(data, model='additive', period=period)
            trend = decomposition.trend.dropna()
            return abs(trend.iloc[-1] - trend.iloc[0]) > 0.1 * data.std()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("Trend detection failed: %s", exc)
            return False
    
    def _detect_seasonality(self, data: pd.Series) -> bool:
        """Detect if data has seasonality"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            if len(data) < 24:
                return False
            period = min(12, max(2, len(data) // 2))
            decomposition = seasonal_decompose(data, model='additive', period=period)
            seasonal = decomposition.seasonal.dropna()
            return seasonal.std() > 0.1 * data.std()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("Seasonality detection failed: %s", exc)
            return False
    
    def _is_stationary(self, data: pd.Series) -> bool:
        """Check if data is stationary using Augmented Dickey-Fuller test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(data.dropna())
            return result[1] < 0.05  # p-value < 0.05 means stationary
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("ADF test failed: %s", exc)
            return False
    
    def create_visualization_data(self, prediction: Prediction) -> Dict[str, Any]:
        """Create data for visualization"""
        try:
            if prediction.status != 'completed':
                raise ValueError("Prediction must be completed to create visualization")
            
            # Get original data
            target_series, _ = self.prepare_data(prediction.dataset)
            
            # Get predictions
            predictions_data = prediction.predictions_data
            forecasts = predictions_data['forecast']
            forecast_dates = predictions_data['forecast_dates']
            
            # Create visualization data
            viz_data = {
                'historical': {
                    'dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) 
                             for date in target_series.index],
                    'values': target_series.tolist()
                },
                'forecast': {
                    'dates': forecast_dates,
                    'values': forecasts
                },
                'metrics': prediction.metrics,
                'model_name': prediction.prediction_model.name
            }
            
            return {
                'success': True,
                'data': viz_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Initialize prediction models in database
def initialize_prediction_models():
    """Initialize prediction models in the database"""
    models_data = [
        {
            'name': 'ARIMA',
            'algorithm_type': 'arima',
            'description': 'Modelo autorregressivo integrado de médias móveis. Ideal para séries temporais com tendência e sazonalidade.',
            'parameters': {'order': '(1,1,1)', 'auto_order': True}
        },
        {
            'name': 'ETS (Exponential Smoothing)',
            'algorithm_type': 'ets',
            'description': 'Suavização exponencial. Excelente para séries com padrões sazonais claros.',
            'parameters': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12}
        },
        {
            'name': 'Prophet',
            'algorithm_type': 'prophet',
            'description': 'Algoritmo do Facebook para previsão de séries temporais. Muito robusto e fácil de usar.',
            'parameters': {'yearly_seasonality': True, 'weekly_seasonality': True, 'daily_seasonality': False}
        },
        {
            'name': 'LSTM',
            'algorithm_type': 'lstm',
            'description': 'Rede neural recorrente de memória longa. Poderoso para padrões complexos.',
            'parameters': {'sequence_length': 10, 'units': 50, 'epochs': 100}
        },
        {
            'name': 'Regressão Linear',
            'algorithm_type': 'linear_regression',
            'description': 'Modelo linear simples. Rápido e eficaz para tendências lineares.',
            'parameters': {}
        },
        {
            'name': 'Regressão Ridge',
            'algorithm_type': 'ridge',
            'description': 'Regressão linear com regularização L2. Evita overfitting.',
            'parameters': {'alpha': 1.0}
        },
        {
            'name': 'Regressão Lasso',
            'algorithm_type': 'lasso',
            'description': 'Regressão linear com regularização L1. Seleciona features automaticamente.',
            'parameters': {'alpha': 1.0}
        },
        {
            'name': 'Random Forest',
            'algorithm_type': 'random_forest',
            'description': 'Ensemble de árvores de decisão. Robusto e não-paramétrico.',
            'parameters': {'n_estimators': 100, 'max_depth': 10}
        },
        {
            'name': 'Support Vector Regression',
            'algorithm_type': 'svr',
            'description': 'Máquinas de vetores de suporte para regressão. Bom para dados não-lineares.',
            'parameters': {'kernel': 'rbf', 'C': 1.0}
        },
        {
            'name': 'Rede Neural',
            'algorithm_type': 'neural_network',
            'description': 'Perceptron multicamadas. Poderoso para padrões complexos.',
            'parameters': {'hidden_layer_sizes': (100, 50), 'max_iter': 1000}
        },
        {
            'name': 'XGBoost',
            'algorithm_type': 'xgboost',
            'description': 'Gradient boosting otimizado. Muito eficaz para competições de ML.',
            'parameters': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        },
        {
            'name': 'LightGBM',
            'algorithm_type': 'lightgbm',
            'description': 'Gradient boosting rápido e eficiente. Bom com muitas features.',
            'parameters': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31}
        },
    ]
    
    for model_data in models_data:
        PredictionModel.objects.get_or_create(
            algorithm_type=model_data['algorithm_type'],
            defaults={
                'name': model_data['name'],
                'description': model_data['description'],
                'parameters': model_data['parameters'],
                'is_active': True
            }
        )
