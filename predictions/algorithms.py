"""
Core algorithms for time series forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import inspect
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine Learning models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Deep Learning (if available)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Prophet (if available)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# XGBoost (if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM (if available)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BaseForecaster:
    """Base class for all forecasting algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.metrics = {}
        self._fit_series: Optional[pd.Series] = None

    def _coerce_params(self, params_obj: Any) -> Dict[str, Any]:
        try:
            # pandas Series
            if hasattr(params_obj, 'to_dict'):
                return {k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v)) for k, v in params_obj.to_dict().items()}  # type: ignore[name-defined]
            # dict
            if isinstance(params_obj, dict):
                return params_obj
            # numpy array with names not available
            try:
                import numpy as np  # local import for safety
                if isinstance(params_obj, (np.ndarray, list, tuple)):
                    return {f'p{i}': float(v) for i, v in enumerate(list(params_obj))}
            except Exception:
                pass
            return {'params': str(params_obj)}
        except Exception:
            return {'params': 'unavailable'}
        
    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit the model to the data"""
        raise NotImplementedError
        
    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        raise NotImplementedError
        
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        # Convert to numpy arrays
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        mse = mean_squared_error(y_true_arr, y_pred_arr)
        mae = mean_absolute_error(y_true_arr, y_pred_arr)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_arr, y_pred_arr)

        # Robust MAPE: ignore zero denominators; fallback to sMAPE
        eps = 1e-8
        denom = np.where(np.abs(y_true_arr) > eps, np.abs(y_true_arr), np.nan)
        mape_vec = np.abs((y_true_arr - y_pred_arr) / denom) * 100.0
        mape = float(np.nanmean(mape_vec)) if np.any(~np.isnan(mape_vec)) else float('nan')

        # sMAPE (always defined unless both are exactly zero)
        smape_den = (np.abs(y_true_arr) + np.abs(y_pred_arr))
        smape_vec = np.where(smape_den > eps, np.abs(y_pred_arr - y_true_arr) / smape_den, 0.0) * 100.0
        smape = float(np.mean(smape_vec))

        if np.isnan(mape):
            mape = smape
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'r2': r2
        }


class ARIMAForecaster(BaseForecaster):
    """ARIMA (AutoRegressive Integrated Moving Average) model"""
    
    def __init__(self, order: Tuple[int, int, int] = None, seasonal_order: Tuple[int, int, int, int] = None):
        super().__init__("ARIMA")
        self.order = order or (1, 1, 1)
        self.seasonal_order = seasonal_order
        self.auto_order = order is None
        self.fitted_model = None
        
    def _find_best_order(self, data: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using AIC"""
        from itertools import product
        
        p_values = range(0, 4)
        d_values = range(0, 3)
        q_values = range(0, 4)
        
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p, d, q in product(p_values, d_values, q_values):
            try:
                model = ARIMA(data, order=(p, d, q))
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = (p, d, q)
            except:
                continue
                
        return best_order
    
    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit ARIMA model"""
        try:
            # Sanitize data
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            # Ensure numeric series
            data = pd.to_numeric(data, errors='coerce').dropna()
            # Ensure sorted by index
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()
            # Use a simple integer index to avoid backend issues comparing index types
            data = pd.Series(data.values, index=pd.RangeIndex(start=0, stop=len(data), step=1), name=data.name)
            self._fit_series = data
            if self.auto_order:
                self.order = self._find_best_order(data)
            
            self.model = ARIMA(data, order=self.order, seasonal_order=self.seasonal_order)
            fitted_model = self.model.fit()
            self.is_fitted = True
            self.fitted_model = fitted_model
            
            return {
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'params': self._coerce_params(getattr(fitted_model, 'params', {}))
            }
        except Exception as e:
            raise ValueError(f"ARIMA fitting failed: {str(e)}")
    
    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Make ARIMA predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        res = self.fitted_model.get_forecast(steps=steps)
        forecast = res.predicted_mean
        return pd.Series(forecast.values if hasattr(forecast, 'values') else forecast, name='forecast')


class ETSForecaster(BaseForecaster):
    """Exponential Smoothing (ETS) model"""
    
    def __init__(self, trend: str = 'add', seasonal: str = 'add', seasonal_periods: int = 12):
        super().__init__("ETS")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        
    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit ETS model"""
        try:
            self._fit_series = data
            self.model = ExponentialSmoothing(
                data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
            fitted_model = self.model.fit()
            self.is_fitted = True
            self.fitted_model = fitted_model
            
            return {
                'trend': self.trend,
                'seasonal': self.seasonal,
                'seasonal_periods': self.seasonal_periods,
                'aic': getattr(fitted_model, 'aic', None),
                'bic': getattr(fitted_model, 'bic', None),
                'params': self._coerce_params(getattr(fitted_model, 'params', {}))
            }
        except Exception as e:
            raise ValueError(f"ETS fitting failed: {str(e)}")
    
    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Make ETS predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        forecast = self.fitted_model.forecast(steps=steps)
        return pd.Series(forecast.values if hasattr(forecast, 'values') else forecast, name='forecast')


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet model"""
    
    def __init__(self, **kwargs):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        super().__init__("Prophet")
        self.prophet_kwargs = kwargs
        
    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit Prophet model"""
        try:
            self._fit_series = data
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
            
            self.model = Prophet(**self.prophet_kwargs)
            self.model.fit(df)
            self.is_fitted = True
            
            return {
                'params': self.prophet_kwargs,
                'components': list(self.model.component_modes.keys())
            }
        except Exception as e:
            raise ValueError(f"Prophet fitting failed: {str(e)}")
    
    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Make Prophet predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        
        # Return only the forecasted values
        forecast_values = forecast['yhat'].tail(steps)
        return pd.Series(forecast_values.values, name='forecast')


class LSTMForecaster(BaseForecaster):
    """LSTM Neural Network model"""
    
    def __init__(self, sequence_length: int = 10, units: int = 50, epochs: int = 100):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install with: pip install tensorflow")
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.scaler = StandardScaler()
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit LSTM model"""
        try:
            self._fit_series = data
            # Prepare data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
            X, y = self._create_sequences(scaled_data)
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build model
            self.model = Sequential([
                LSTM(self.units, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(0.2),
                LSTM(self.units, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            history = self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
            self.is_fitted = True
            
            return {
                'sequence_length': self.sequence_length,
                'units': self.units,
                'epochs': self.epochs,
                'final_loss': history.history['loss'][-1]
            }
        except Exception as e:
            raise ValueError(f"LSTM fitting failed: {str(e)}")
    
    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Make LSTM predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        if self._fit_series is None or len(self._fit_series) == 0:
            raise ValueError("No fitted series available for prediction")
        # Get last sequence
        last_sequence = self.scaler.transform(self._fit_series.tail(self.sequence_length).values.reshape(-1, 1)).flatten()
        last_sequence = last_sequence.reshape((1, self.sequence_length, 1))
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return pd.Series(predictions, name='forecast')


class MLForecaster(BaseForecaster):
    """Machine Learning models for time series"""
    
    def __init__(self, model_type: str, **kwargs):
        super().__init__(model_type)
        self.model_type = model_type
        self.kwargs = kwargs
        self.scaler = StandardScaler()
    
    def _filtered_kwargs(self, estimator_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sig = inspect.signature(estimator_cls.__init__)
            allowed = set(sig.parameters.keys())
            # Remove 'self'
            allowed.discard('self')
            return {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            # Fallback: remove known non-estimator keys
            blacklist = {'frequency'}
            return {k: v for k, v in kwargs.items() if k not in blacklist}
        
    def _create_features(self, data: pd.Series, lags: int = 10) -> pd.DataFrame:
        """Create comprehensive time series features for ML models"""
        # Ensure the series has a name for column operations
        series_name = data.name or 'target'
        if data.name != series_name:
            data = data.rename(series_name)
        df = pd.DataFrame(data)
        
        # Multiple lag features (more comprehensive)
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = data.shift(i)
        
        # Multiple rolling windows for different time horizons
        for window in [3, 7, 14, 30]:
            df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = data.rolling(window=window).std()
            df[f'rolling_max_{window}'] = data.rolling(window=window).max()
            df[f'rolling_min_{window}'] = data.rolling(window=window).min()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f'ema_{alpha}'] = data.ewm(alpha=alpha).mean()
        
        # Trend features
        df['diff_1'] = data.diff(1)  # First difference
        df['diff_2'] = data.diff(2)  # Second difference
        df['pct_change'] = data.pct_change()
        
        # Volatility features
        df['volatility_5'] = data.rolling(window=5).std()
        df['volatility_20'] = data.rolling(window=20).std()
        
        # Time-based features (if datetime index)
        if hasattr(data.index, 'month'):
            df['month'] = data.index.month
            df['quarter'] = data.index.quarter
            df['day_of_year'] = data.index.dayofyear
            df['day_of_week'] = data.index.dayofweek
            df['is_month_end'] = data.index.is_month_end.astype(int)
            df['is_quarter_end'] = data.index.is_quarter_end.astype(int)
            df['is_year_end'] = data.index.is_year_end.astype(int)
        
        # Polynomial features for non-linear relationships
        if len(data) > 10:
            df['squared'] = data ** 2
            df['sqrt'] = np.sqrt(np.abs(data))
        
        return df.dropna()
    
    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit ML model"""
        try:
            # Ensure named series
            series_name = data.name or 'target'
            if data.name != series_name:
                data = data.rename(series_name)
            self._fit_series = data
            # Create features
            feature_df = self._create_features(data)
            # Guard against empty feature set after dropna/lagging
            if feature_df.shape[0] == 0:
                raise ValueError("Insufficient samples after feature engineering. Provide more historical data.")
            X = feature_df.drop(columns=[series_name])
            y = feature_df[series_name]
            
            # Sanitize features: replace inf/nan with finite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())  # Fill remaining NaN with median
            
            # Check for any remaining infinite values
            if np.isinf(X.values).any():
                X = X.replace([np.inf, -np.inf], 0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize model
            if self.model_type == 'linear_regression':
                self.model = LinearRegression(**self._filtered_kwargs(LinearRegression, self.kwargs))
            elif self.model_type == 'ridge':
                self.model = Ridge(**self._filtered_kwargs(Ridge, self.kwargs))
            elif self.model_type == 'lasso':
                self.model = Lasso(**self._filtered_kwargs(Lasso, self.kwargs))
            elif self.model_type == 'random_forest':
                self.model = RandomForestRegressor(**self._filtered_kwargs(RandomForestRegressor, self.kwargs))
            elif self.model_type == 'svr':
                self.model = SVR(**self._filtered_kwargs(SVR, self.kwargs))
            elif self.model_type == 'neural_network':
                self.model = MLPRegressor(**self._filtered_kwargs(MLPRegressor, self.kwargs))
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            return {
                'model_type': self.model_type,
                'n_features': X.shape[1],
                'n_samples': len(y)
            }
        except Exception as e:
            raise ValueError(f"ML model fitting failed: {str(e)}")
    
    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Make ML predictions with improved time series forecasting"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        if self._fit_series is None or len(self._fit_series) == 0:
            raise ValueError("No fitted series available for prediction")
        
        predictions = []
        # Use more historical data for better context (at least 30 points)
        min_context = min(30, len(self._fit_series))
        last_data = self._fit_series.tail(min_context)
        
        for step in range(steps):
            # Create features for next prediction
            feature_df = self._create_features(last_data)
            
            if feature_df.empty:
                # Fallback: use trend-based naive forecast
                if len(last_data) >= 2:
                    trend = last_data.iloc[-1] - last_data.iloc[-2]
                    pred = float(last_data.iloc[-1] + trend)
                else:
                    pred = float(last_data.iloc[-1])
            else:
                X = feature_df.drop(columns=[last_data.name]).iloc[-1:]
                X_scaled = self.scaler.transform(X)
                pred = float(self.model.predict(X_scaled)[0])
                
                # Add some randomness to prevent identical predictions
                # Use a small amount of noise based on historical volatility
                if len(last_data) > 5:
                    volatility = last_data.rolling(window=5).std().iloc[-1]
                    if not pd.isna(volatility) and volatility > 0:
                        noise = np.random.normal(0, volatility * 0.1)
                        pred += noise
            
            # Apply non-negative constraint if needed (after all processing)
            if len(last_data) > 0 and (last_data >= 0).sum() / len(last_data) > 0.8:
                pred = max(0, pred)
            
            predictions.append(pred)
            
            # Update data for next iteration with proper index handling
            next_index = last_data.index[-1]
            try:
                if hasattr(next_index, 'to_pydatetime'):
                    # Datetime index
                    if hasattr(next_index, 'day'):
                        next_index = next_index + pd.Timedelta(days=1)
                    else:
                        next_index = next_index + 1
                else:
                    # Numeric index
                    next_index = next_index + 1
            except Exception:
                # Fallback for any index type
                next_index = len(last_data)
            
            new_point = pd.Series([pred], index=[next_index], name=last_data.name)
            last_data = pd.concat([last_data, new_point]).tail(min_context)
        
        return pd.Series(predictions, name='forecast')


class XGBoostForecaster(BaseForecaster):
    """XGBoost model for time series"""
    
    def __init__(self, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        super().__init__("XGBoost")
        self.kwargs = kwargs
        self.scaler = StandardScaler()
        
    def _create_features(self, data: pd.Series, lags: int = 15) -> pd.DataFrame:
        """Create comprehensive features for XGBoost"""
        df = pd.DataFrame(data)
        
        # Multiple lag features (more comprehensive)
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = data.shift(i)
        
        # Multiple rolling windows for different time horizons
        for window in [3, 7, 14, 30, 60]:
            df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = data.rolling(window=window).std()
            df[f'rolling_max_{window}'] = data.rolling(window=window).max()
            df[f'rolling_min_{window}'] = data.rolling(window=window).min()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            df[f'ema_{alpha}'] = data.ewm(alpha=alpha).mean()
        
        # Trend and momentum features
        df['diff_1'] = data.diff(1)
        df['diff_2'] = data.diff(2)
        df['pct_change'] = data.pct_change()
        df['momentum_5'] = data / data.shift(5).replace(0, np.nan) - 1
        df['momentum_10'] = data / data.shift(10).replace(0, np.nan) - 1
        
        # Volatility features
        df['volatility_5'] = data.rolling(window=5).std()
        df['volatility_20'] = data.rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20'].replace(0, np.nan)
        
        # Time-based features (if datetime index)
        if hasattr(data.index, 'month'):
            df['month'] = data.index.month
            df['quarter'] = data.index.quarter
            df['day_of_year'] = data.index.dayofyear
            df['day_of_week'] = data.index.dayofweek
            df['is_month_end'] = data.index.is_month_end.astype(int)
            df['is_quarter_end'] = data.index.is_quarter_end.astype(int)
            df['is_year_end'] = data.index.is_year_end.astype(int)
        
        # Interaction features
        if len(data) > 20:
            df['lag_1_x_rolling_mean_7'] = df['lag_1'] * df['rolling_mean_7']
            df['lag_1_x_volatility_5'] = df['lag_1'] * df['volatility_5']
        
        return df.dropna()
    
    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit XGBoost model"""
        try:
            # Store the original series for prediction
            self._fit_series = data
            
            # Create features
            feature_df = self._create_features(data)
            X = feature_df.drop(columns=[data.name])
            y = feature_df[data.name]
            
            # Sanitize features: replace inf/nan with finite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())  # Fill remaining NaN with median
            
            # Check for any remaining infinite values
            if np.isinf(X.values).any():
                X = X.replace([np.inf, -np.inf], 0)
            
            # Initialize model
            self.model = xgb.XGBRegressor(**self.kwargs)
            
            # Train model
            self.model.fit(X, y)
            self.is_fitted = True
            
            return {
                'n_features': X.shape[1],
                'n_samples': len(y),
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
            }
        except Exception as e:
            raise ValueError(f"XGBoost fitting failed: {str(e)}")
    
    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Make XGBoost predictions with improved time series forecasting"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        if self._fit_series is None or len(self._fit_series) == 0:
            raise ValueError("No fitted series available for prediction")
        
        predictions = []
        # Use more historical data for better context (at least 50 points for XGBoost)
        min_context = min(50, len(self._fit_series))
        last_data = self._fit_series.tail(min_context)
        
        for step in range(steps):
            feature_df = self._create_features(last_data)
            
            if feature_df.empty:
                # Fallback: use trend-based naive forecast
                if len(last_data) >= 3:
                    recent_trend = (last_data.iloc[-1] - last_data.iloc[-3]) / 2
                    pred = float(last_data.iloc[-1] + recent_trend)
                else:
                    pred = float(last_data.iloc[-1])
            else:
                X = feature_df.drop(columns=[last_data.name]).iloc[-1:]
                pred = float(self.model.predict(X)[0])
                
                # Add adaptive noise based on feature importance and volatility
                if len(last_data) > 10:
                    volatility = last_data.rolling(window=10).std().iloc[-1]
                    if not pd.isna(volatility) and volatility > 0:
                        # Use smaller noise for XGBoost as it's more sophisticated
                        noise = np.random.normal(0, volatility * 0.05)
                        pred += noise
            
            # Apply non-negative constraint if needed (after all processing)
            if len(last_data) > 0 and (last_data >= 0).sum() / len(last_data) > 0.8:
                pred = max(0, pred)
            
            predictions.append(pred)
            
            # Update data for next iteration with proper index handling
            next_index = last_data.index[-1]
            try:
                if hasattr(next_index, 'to_pydatetime'):
                    # Datetime index
                    if hasattr(next_index, 'day'):
                        next_index = next_index + pd.Timedelta(days=1)
                    else:
                        next_index = next_index + 1
                else:
                    # Numeric index
                    next_index = next_index + 1
            except Exception:
                # Fallback for any index type
                next_index = len(last_data)
            
            new_point = pd.Series([pred], index=[next_index], name=last_data.name)
            last_data = pd.concat([last_data, new_point]).tail(min_context)
        
        return pd.Series(predictions, name='forecast')


class LightGBMForecaster(BaseForecaster):
    """LightGBM model for time series"""
    def __init__(self, **kwargs):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        super().__init__("LightGBM")
        self.kwargs = kwargs
        self.scaler = StandardScaler()

    def _create_features(self, data: pd.Series, lags: int = 15) -> pd.DataFrame:
        """Create comprehensive features for LightGBM"""
        df = pd.DataFrame(data)
        
        # Multiple lag features (more comprehensive)
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = data.shift(i)
        
        # Multiple rolling windows for different time horizons
        for window in [3, 7, 14, 30, 60]:
            df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = data.rolling(window=window).std()
            df[f'rolling_max_{window}'] = data.rolling(window=window).max()
            df[f'rolling_min_{window}'] = data.rolling(window=window).min()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            df[f'ema_{alpha}'] = data.ewm(alpha=alpha).mean()
        
        # Trend and momentum features
        df['diff_1'] = data.diff(1)
        df['diff_2'] = data.diff(2)
        df['pct_change'] = data.pct_change()
        df['momentum_5'] = data / data.shift(5).replace(0, np.nan) - 1
        df['momentum_10'] = data / data.shift(10).replace(0, np.nan) - 1
        
        # Volatility features
        df['volatility_5'] = data.rolling(window=5).std()
        df['volatility_20'] = data.rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20'].replace(0, np.nan)
        
        # Time-based features (if datetime index)
        if hasattr(data.index, 'month'):
            df['month'] = data.index.month
            df['quarter'] = data.index.quarter
            df['day_of_year'] = data.index.dayofyear
            df['day_of_week'] = data.index.dayofweek
            df['is_month_end'] = data.index.is_month_end.astype(int)
            df['is_quarter_end'] = data.index.is_quarter_end.astype(int)
            df['is_year_end'] = data.index.is_year_end.astype(int)
        
        # Interaction features
        if len(data) > 20:
            df['lag_1_x_rolling_mean_7'] = df['lag_1'] * df['rolling_mean_7']
            df['lag_1_x_volatility_5'] = df['lag_1'] * df['volatility_5']
        
        return df.dropna()

    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        try:
            # Store the original series for prediction
            self._fit_series = data
            
            feature_df = self._create_features(data)
            X = feature_df.drop(columns=[data.name])
            y = feature_df[data.name]
            
            # Sanitize features: replace inf/nan with finite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())  # Fill remaining NaN with median
            
            # Check for any remaining infinite values
            if np.isinf(X.values).any():
                X = X.replace([np.inf, -np.inf], 0)
            
            self.model = lgb.LGBMRegressor(**self.kwargs)
            self.model.fit(X, y)
            self.is_fitted = True
            importances = dict(zip(X.columns, self.model.feature_importances_.tolist()))
            return {
                'n_features': X.shape[1],
                'n_samples': len(y),
                'feature_importance': importances
            }
        except Exception as e:
            raise ValueError(f"LightGBM fitting failed: {str(e)}")

    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Make LightGBM predictions with improved time series forecasting"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        if self._fit_series is None or len(self._fit_series) == 0:
            raise ValueError("No fitted series available for prediction")
        
        predictions = []
        # Use more historical data for better context (at least 50 points for LightGBM)
        min_context = min(50, len(self._fit_series))
        last_data = self._fit_series.tail(min_context)
        
        for step in range(steps):
            feature_df = self._create_features(last_data)
            
            if feature_df.empty:
                # Fallback: use trend-based naive forecast
                if len(last_data) >= 3:
                    recent_trend = (last_data.iloc[-1] - last_data.iloc[-3]) / 2
                    pred = float(last_data.iloc[-1] + recent_trend)
                else:
                    pred = float(last_data.iloc[-1])
            else:
                X = feature_df.drop(columns=[last_data.name]).iloc[-1:]
                pred = float(self.model.predict(X)[0])
                
                # Add adaptive noise based on volatility
                if len(last_data) > 10:
                    volatility = last_data.rolling(window=10).std().iloc[-1]
                    if not pd.isna(volatility) and volatility > 0:
                        # Use smaller noise for LightGBM as it's more sophisticated
                        noise = np.random.normal(0, volatility * 0.05)
                        pred += noise
            
            # Apply non-negative constraint if needed (after all processing)
            if len(last_data) > 0 and (last_data >= 0).sum() / len(last_data) > 0.8:
                pred = max(0, pred)
            
            predictions.append(pred)
            
            # Update data for next iteration with proper index handling
            next_index = last_data.index[-1]
            try:
                if hasattr(next_index, 'to_pydatetime'):
                    # Datetime index
                    if hasattr(next_index, 'day'):
                        next_index = next_index + pd.Timedelta(days=1)
                    else:
                        next_index = next_index + 1
                else:
                    # Numeric index
                    next_index = next_index + 1
            except Exception:
                # Fallback for any index type
                next_index = len(last_data)
            
            new_point = pd.Series([pred], index=[next_index], name=last_data.name)
            last_data = pd.concat([last_data, new_point]).tail(min_context)
        
        return pd.Series(predictions, name='forecast')


class ModelEvaluator:
    """Evaluate and compare forecasting models"""
    
    @staticmethod
    def evaluate_model(model: BaseForecaster, train_data: pd.Series, test_data: pd.Series) -> Dict[str, Any]:
        """Evaluate a single model"""
        # Fit model
        model.fit(train_data)
        
        # Make predictions
        predictions = model.predict(len(test_data))
        
        # Calculate metrics
        metrics = model.evaluate(test_data, predictions)
        
        return {
            'model_name': model.name,
            'metrics': metrics,
            'predictions': predictions,
            'is_fitted': model.is_fitted
        }
    
    @staticmethod
    def compare_models(models: List[BaseForecaster], train_data: pd.Series, test_data: pd.Series) -> pd.DataFrame:
        """Compare multiple models"""
        results = []
        
        for model in models:
            try:
                result = ModelEvaluator.evaluate_model(model, train_data, test_data)
                results.append({
                    'Model': result['model_name'],
                    'RMSE': result['metrics']['rmse'],
                    'MAE': result['metrics']['mae'],
                    'MAPE': result['metrics']['mape'],
                    'R²': result['metrics']['r2']
                })
            except Exception as e:
                results.append({
                    'Model': model.name,
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'MAPE': float('inf'),
                    'R²': -float('inf'),
                    'Error': str(e)
                })
        
        return pd.DataFrame(results).sort_values('RMSE')


# Factory function to create models
def create_forecaster(model_type: str, **kwargs) -> BaseForecaster:
    """Create a forecaster instance"""
    model_map = {
        'arima': ARIMAForecaster,
        'ets': ETSForecaster,
        'prophet': ProphetForecaster,
        'lstm': LSTMForecaster,
        'linear_regression': lambda **k: MLForecaster('linear_regression', **k),
        'ridge': lambda **k: MLForecaster('ridge', **k),
        'lasso': lambda **k: MLForecaster('lasso', **k),
        'random_forest': lambda **k: MLForecaster('random_forest', **k),
        'svr': lambda **k: MLForecaster('svr', **k),
        'neural_network': lambda **k: MLForecaster('neural_network', **k),
        'xgboost': XGBoostForecaster,
        'lightgbm': LightGBMForecaster,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    ctor = model_map[model_type]
    # Harmonize & coerce params
    if model_type == 'ets':
        allowed = {'trend', 'seasonal', 'seasonal_periods'}
        coerced: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == 'seasonal_periods' and isinstance(v, str):
                try:
                    coerced[k] = int(v)
                except Exception:
                    coerced[k] = v
            else:
                coerced[k] = v
        kwargs = coerced
    if model_type == 'arima':
        allowed = {'order', 'seasonal_order'}
        coerced: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if isinstance(v, str):
                s = v.strip().strip('()').strip('[]')
                parts = [p for p in s.split(',') if p.strip() != '']
                try:
                    tuple_val = tuple(int(p) for p in parts)
                    coerced[k] = tuple_val
                except Exception:
                    coerced[k] = v
            elif isinstance(v, list):
                try:
                    coerced[k] = tuple(int(p) for p in v)
                except Exception:
                    coerced[k] = tuple(v)
            else:
                coerced[k] = v
        kwargs = coerced
    return ctor(**kwargs)
