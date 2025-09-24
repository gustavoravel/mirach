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
    
    def prepare_data(self, dataset: Dataset) -> Tuple[pd.Series, List[str]]:
        """Prepare data from dataset for forecasting"""
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
            
            for mapping in mappings:
                if mapping.column_type == 'timestamp':
                    timestamp_col = mapping.column_name
                elif mapping.column_type == 'target':
                    target_col = mapping.column_name
                elif mapping.column_type == 'feature':
                    feature_cols.append(mapping.column_name)
            
            if not timestamp_col or not target_col:
                raise ValueError("Timestamp and target columns must be mapped")
            
            # Convert timestamp column
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            df = df.set_index(timestamp_col)
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Get target series
            # Coerce target to numeric to avoid string comparison errors in models
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            target_series = df[target_col].dropna()
            
            if len(target_series) == 0:
                raise ValueError(f"Target column '{target_col}' has no valid numeric data after conversion")
            
            return target_series, feature_cols
            
        except Exception as e:
            raise ValueError(f"Data preparation failed: {str(e)}")
    
    def split_data(self, data: pd.Series, train_size: float = 0.8, 
                   validation_size: float = 0.1) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets"""
        n = len(data)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + validation_size))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    def run_prediction(self, prediction: Prediction) -> Dict[str, Any]:
        """Run prediction using the specified model"""
        try:
            # Update status
            prediction.status = 'training'
            prediction.save()
            
            # Prepare data
            target_series, feature_cols = self.prepare_data(prediction.dataset)
            
            # Split data
            train_data, val_data, test_data = self.split_data(
                target_series, 
                prediction.train_size, 
                prediction.validation_size
            )
            
            # Create forecaster
            forecaster = create_forecaster(
                prediction.prediction_model.algorithm_type,
                **prediction.model_parameters
            )

            # Validate minimum samples for ML models (lags/rolling drop initial rows)
            ml_models = {'linear_regression','ridge','lasso','random_forest','svr','neural_network','xgboost','lightgbm'}
            if prediction.prediction_model.algorithm_type in ml_models:
                if len(train_data) < 15:
                    raise ValueError("Insufficient training samples for ML model. Provide at least ~15-20 points.")
            
            # Fit model
            fit_results = forecaster.fit(train_data)
            
            # Make predictions
            predictions = forecaster.predict(prediction.prediction_horizon)
            
            # Evaluate on validation set if available
            if len(val_data) > 0:
                val_predictions = forecaster.predict(len(val_data))
                metrics = forecaster.evaluate(val_data, val_predictions)
            else:
                # Quick naive baseline for metrics when no val set
                metrics = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0}
            
            # Explainability placeholder (depends on model)
            explanation = {}
            try:
                if hasattr(forecaster, 'feature_importances_'):
                    explanation['feature_importances'] = getattr(forecaster, 'feature_importances_', None)
                if hasattr(forecaster, 'coef_'):
                    explanation['coefficients'] = getattr(forecaster, 'coef_', None)
            except Exception:
                pass

            # Sanitize metrics (replace NaN/Inf with None for JSON)
            metrics = self._sanitize_json(metrics)

            # Create prediction results
            self._create_prediction_results(prediction, predictions, target_series, prediction.model_parameters.get('frequency'))
            
            # Update prediction
            prediction.status = 'completed'
            prediction.completed_at = datetime.now()
            prediction.metrics = metrics
            prediction.predictions_data = self._sanitize_json({
                'forecast': predictions.tolist(),
                'forecast_dates': self._generate_forecast_dates(
                    target_series.index[-1], 
                    prediction.prediction_horizon,
                    prediction.model_parameters.get('frequency')
                ),
                'fit_results': fit_results
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
            prediction.explainability = explanation
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
                                 frequency: Optional[str] = None):
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
            PredictionResult.objects.create(
                prediction=prediction,
                timestamp=date,
                predicted_value=float(forecast_value)
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
            target_series, _ = self.prepare_data(dataset)
            
            # Split data
            train_data, val_data, test_data = self.split_data(target_series, train_size)
            
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
        """Get model recommendations based on data characteristics"""
        try:
            target_series, _ = self.prepare_data(dataset)
            
            recommendations = []
            
            # Analyze data characteristics
            data_length = len(target_series)
            has_trend = self._detect_trend(target_series)
            has_seasonality = self._detect_seasonality(target_series)
            is_stationary = self._is_stationary(target_series)
            
            # Recommend models based on characteristics
            if data_length < 50:
                recommendations.extend(['linear_regression', 'ridge', 'lasso'])
            elif data_length < 200:
                recommendations.extend(['arima', 'ets', 'linear_regression', 'ridge'])
            else:
                recommendations.extend(['arima', 'ets', 'prophet', 'lstm', 'xgboost'])
            
            if has_seasonality:
                recommendations.extend(['ets', 'prophet'])
            
            if not is_stationary:
                recommendations.extend(['arima', 'prophet'])
            
            # Remove duplicates and limit to top 5
            recommendations = list(dict.fromkeys(recommendations))[:5]
            
            return {
                'success': True,
                'recommendations': recommendations,
                'data_characteristics': {
                    'length': data_length,
                    'has_trend': has_trend,
                    'has_seasonality': has_seasonality,
                    'is_stationary': is_stationary
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
            decomposition = seasonal_decompose(data, model='additive', period=min(12, len(data)//2))
            trend = decomposition.trend.dropna()
            return abs(trend.iloc[-1] - trend.iloc[0]) > 0.1 * data.std()
        except:
            return False
    
    def _detect_seasonality(self, data: pd.Series) -> bool:
        """Detect if data has seasonality"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            if len(data) < 24:
                return False
            decomposition = seasonal_decompose(data, model='additive', period=min(12, len(data)//2))
            seasonal = decomposition.seasonal.dropna()
            return seasonal.std() > 0.1 * data.std()
        except:
            return False
    
    def _is_stationary(self, data: pd.Series) -> bool:
        """Check if data is stationary using Augmented Dickey-Fuller test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(data.dropna())
            return result[1] < 0.05  # p-value < 0.05 means stationary
        except:
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
        }
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
