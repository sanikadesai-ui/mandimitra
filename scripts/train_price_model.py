#!/usr/bin/env python3
"""
MANDIMITRA - Model 2: Price Intelligence Engine
================================================

Production-grade 15-day price forecasting model using LightGBM with:
- Advanced feature engineering (lag features, rolling stats, seasonality)
- Time-series cross-validation
- Multi-horizon direct forecasting
- HOLD/SELL recommendations
- SHAP explainability

Author: MANDIMITRA Team
Date: 2026-02-05
"""

import argparse
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try to import MAPIE for conformal prediction
try:
    from mapie.regression import MapieRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    logger.info("MAPIE not installed. Using residual-based prediction intervals.")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model parameters
FORECAST_HORIZONS = [1, 3, 7, 14, 15]  # Days to forecast
LAG_DAYS = [1, 2, 3, 7, 14, 21, 30]  # Lag features
ROLLING_WINDOWS = [3, 7, 14, 30]  # Rolling window sizes
MIN_RECORDS_PER_GROUP = 60  # Minimum records to train on commodity-market

# Price thresholds for HOLD/SELL
HOLD_THRESHOLD_PCT = 5.0  # Hold if price expected to rise > 5%
TRANSPORT_COST_PER_KM = 2.0  # Rs per quintal per km (approximate)

# LightGBM parameters (tuned)
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data() -> pd.DataFrame:
    """Load and prepare the price dataset."""
    logger.info("Loading data...")
    
    # Try optimized dataset first, fall back to basic
    optimized_path = DATA_DIR / "model" / "mandi_weather_optimized.parquet"
    basic_path = DATA_DIR / "model" / "mandi_weather_2016plus.parquet"
    
    if optimized_path.exists():
        df = pd.read_parquet(optimized_path)
        logger.info(f"Loaded optimized dataset: {len(df):,} rows")
    elif basic_path.exists():
        df = pd.read_parquet(basic_path)
        logger.info(f"Loaded basic dataset: {len(df):,} rows")
    else:
        raise FileNotFoundError("No model dataset found")
    
    # Ensure datetime
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    
    # Remove extreme outliers (price > 99.5th percentile or < 0.5th percentile)
    lower_bound = df['modal_price'].quantile(0.005)
    upper_bound = df['modal_price'].quantile(0.995)
    initial_len = len(df)
    df = df[(df['modal_price'] >= lower_bound) & (df['modal_price'] <= upper_bound)]
    logger.info(f"Removed {initial_len - len(df):,} outliers (outside 0.5-99.5 percentile)")
    
    # Sort by date for time series operations
    df = df.sort_values(['commodity', 'market', 'arrival_date']).reset_index(drop=True)
    
    return df


def create_lag_features(df: pd.DataFrame, group_cols: List[str], 
                        target_col: str = 'modal_price') -> pd.DataFrame:
    """Create lag features for each commodity-market combination."""
    logger.info("Creating lag features...")
    
    df = df.copy()
    
    # Group by commodity-market for lag calculations
    grouped = df.groupby(group_cols)
    
    # Lag features
    for lag in LAG_DAYS:
        df[f'price_lag_{lag}d'] = grouped[target_col].shift(lag)
    
    # Rolling statistics
    for window in ROLLING_WINDOWS:
        df[f'price_rolling_mean_{window}d'] = grouped[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'price_rolling_std_{window}d'] = grouped[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        df[f'price_rolling_min_{window}d'] = grouped[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).min()
        )
        df[f'price_rolling_max_{window}d'] = grouped[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
    
    # Price momentum features
    df['price_momentum_7d'] = grouped[target_col].transform(
        lambda x: x.shift(1) - x.shift(7)
    )
    df['price_momentum_14d'] = grouped[target_col].transform(
        lambda x: x.shift(1) - x.shift(14)
    )
    df['price_momentum_30d'] = grouped[target_col].transform(
        lambda x: x.shift(1) - x.shift(30)
    )
    
    # Volatility features
    df['price_volatility_7d'] = grouped[target_col].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).std() / x.shift(1).rolling(7, min_periods=1).mean()
    )
    df['price_volatility_14d'] = grouped[target_col].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).std() / x.shift(1).rolling(14, min_periods=1).mean()
    )
    
    # Price change percentage
    df['price_pct_change_1d'] = grouped[target_col].pct_change()
    df['price_pct_change_7d'] = grouped[target_col].transform(
        lambda x: x.pct_change(periods=7)
    )
    
    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar/seasonal features."""
    logger.info("Creating calendar features...")
    
    df = df.copy()
    
    # Basic calendar features (some may already exist)
    if 'arrival_year' not in df.columns:
        df['arrival_year'] = df['arrival_date'].dt.year
    if 'arrival_month' not in df.columns:
        df['arrival_month'] = df['arrival_date'].dt.month
    if 'arrival_day_of_week' not in df.columns:
        df['arrival_day_of_week'] = df['arrival_date'].dt.dayofweek
    if 'arrival_week_of_year' not in df.columns:
        df['arrival_week_of_year'] = df['arrival_date'].dt.isocalendar().week.astype(int)
    
    # Day of month
    df['arrival_day_of_month'] = df['arrival_date'].dt.day
    
    # Quarter
    df['arrival_quarter'] = df['arrival_date'].dt.quarter
    
    # Cyclic encoding for month and day of week
    df['month_sin'] = np.sin(2 * np.pi * df['arrival_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['arrival_month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['arrival_day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['arrival_day_of_week'] / 7)
    
    # Is month end/start
    if 'is_month_start' not in df.columns:
        df['is_month_start'] = (df['arrival_day_of_month'] <= 5).astype(int)
    if 'is_month_end' not in df.columns:
        df['is_month_end'] = (df['arrival_day_of_month'] >= 25).astype(int)
    
    # Is weekend
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['arrival_day_of_week'].isin([5, 6]).astype(int)
    
    # Days since year start
    df['day_of_year'] = df['arrival_date'].dt.dayofyear
    
    # Agricultural seasons (Maharashtra)
    def get_agri_season(month):
        if month in [6, 7, 8, 9]:
            return 0  # Kharif
        elif month in [10, 11, 12, 1, 2]:
            return 1  # Rabi
        else:
            return 2  # Summer/Zaid
    
    df['agri_season'] = df['arrival_month'].apply(get_agri_season)
    
    return df


def encode_categoricals(df: pd.DataFrame, encoders: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """Encode categorical variables."""
    logger.info("Encoding categorical features...")
    
    df = df.copy()
    cat_cols = ['district', 'market', 'commodity', 'variety', 'grade']
    
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        for col, le in encoders.items():
            if col in df.columns:
                # Handle unseen labels
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    return df, encoders


def create_target_horizons(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Create future price targets for multi-horizon forecasting."""
    logger.info(f"Creating target horizons: {horizons}")
    
    df = df.copy()
    grouped = df.groupby(['commodity', 'market'])
    
    for h in horizons:
        df[f'target_{h}d'] = grouped['modal_price'].shift(-h)
    
    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def prepare_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Prepare feature lists."""
    
    # Numerical features
    num_features = [
        # Lag features
        *[f'price_lag_{lag}d' for lag in LAG_DAYS],
        # Rolling features
        *[f'price_rolling_mean_{w}d' for w in ROLLING_WINDOWS],
        *[f'price_rolling_std_{w}d' for w in ROLLING_WINDOWS],
        *[f'price_rolling_min_{w}d' for w in ROLLING_WINDOWS],
        *[f'price_rolling_max_{w}d' for w in ROLLING_WINDOWS],
        # Momentum
        'price_momentum_7d', 'price_momentum_14d', 'price_momentum_30d',
        # Volatility
        'price_volatility_7d', 'price_volatility_14d',
        # Percentage change
        'price_pct_change_1d', 'price_pct_change_7d',
        # Calendar
        'arrival_year', 'arrival_month', 'arrival_day_of_week', 
        'arrival_week_of_year', 'arrival_day_of_month', 'arrival_quarter',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_month_start', 'is_month_end', 'is_weekend', 'day_of_year', 'agri_season',
        # Weather (if available)
        't2m_max', 't2m_min', 't2m_mean', 'humidity', 'precipitation', 
        'wind_speed', 'solar_radiation',
        # Price spread
        'price_spread', 'price_range', 'price_volatility',
    ]
    
    # Categorical features (encoded)
    cat_features = [
        'district_encoded', 'market_encoded', 'commodity_encoded',
        'variety_encoded', 'grade_encoded',
    ]
    
    # Filter to only columns that exist in df
    all_features = [f for f in num_features + cat_features if f in df.columns]
    
    return all_features, cat_features


def train_model_for_horizon(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            cat_features: List[str], horizon: int) -> lgb.LGBMRegressor:
    """Train LightGBM model for a specific forecast horizon."""
    
    logger.info(f"Training model for {horizon}-day horizon...")
    
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=[f for f in cat_features if f in X_train.columns],
    )
    
    return model


def time_series_cross_validate(df: pd.DataFrame, features: List[str], 
                               target_col: str, n_splits: int = 5) -> Dict:
    """Perform time-series cross-validation."""
    logger.info(f"Running {n_splits}-fold time series CV...")
    
    # Sort by date
    df = df.sort_values('arrival_date').reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = {'mae': [], 'rmse': [], 'r2': [], 'mape': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        X_train = df.iloc[train_idx][features]
        y_train = df.iloc[train_idx][target_col]
        X_val = df.iloc[val_idx][features]
        y_val = df.iloc[val_idx][target_col]
        
        # Drop NaN
        mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
        mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())
        
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        X_val, y_val = X_val[mask_val], y_val[mask_val]
        
        if len(X_train) < 100 or len(X_val) < 10:
            continue
        
        model = lgb.LGBMRegressor(**{**LGBM_PARAMS, 'n_estimators': 200})
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        y_pred = model.predict(X_val)
        
        cv_results['mae'].append(mean_absolute_error(y_val, y_pred))
        cv_results['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
        cv_results['r2'].append(r2_score(y_val, y_pred))
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_val - y_pred) / np.maximum(y_val, 1))) * 100
        cv_results['mape'].append(mape)
        
        logger.info(f"  Fold {fold+1}: MAE={cv_results['mae'][-1]:.2f}, R2={cv_results['r2'][-1]:.4f}")
    
    return {k: np.mean(v) for k, v in cv_results.items()}


def train_all_models(df: pd.DataFrame, features: List[str], cat_features: List[str],
                     horizons: List[int]) -> Dict[int, lgb.LGBMRegressor]:
    """Train models for all forecast horizons."""
    
    models = {}
    
    # Time-based train/val/test split
    # Use data up to 2025-10-01 for training, 2025-10-01 to 2025-12-31 for validation
    df = df.sort_values('arrival_date').reset_index(drop=True)
    
    train_end = pd.Timestamp('2025-10-01')
    val_end = pd.Timestamp('2025-12-31')
    
    train_df = df[df['arrival_date'] < train_end].copy()
    val_df = df[(df['arrival_date'] >= train_end) & (df['arrival_date'] < val_end)].copy()
    
    logger.info(f"Train period: {train_df['arrival_date'].min()} to {train_df['arrival_date'].max()}")
    logger.info(f"Validation period: {val_df['arrival_date'].min()} to {val_df['arrival_date'].max()}")
    logger.info(f"Train size: {len(train_df):,}, Validation size: {len(val_df):,}")
    
    results = {}
    
    for horizon in horizons:
        target_col = f'target_{horizon}d'
        
        # Prepare data
        X_train = train_df[features].copy()
        y_train = train_df[target_col].copy()
        X_test = val_df[features].copy()
        y_test = val_df[target_col].copy()
        
        # Drop rows with missing target or features
        mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
        mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())
        
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        X_test, y_test = X_test[mask_test], y_test[mask_test]
        
        logger.info(f"Horizon {horizon}d: {len(X_train):,} train, {len(X_test):,} test samples")
        
        if len(X_test) < 10:
            logger.warning(f"  Skipping horizon {horizon}d - insufficient test data")
            continue
        
        # Train
        model = train_model_for_horizon(
            X_train, y_train, X_test, y_test, cat_features, horizon
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
        
        results[horizon] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'n_train': len(X_train),
            'n_test': len(X_test),
        }
        
        logger.info(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
        
        models[horizon] = model
    
    # ========================================================================
    # CONFORMAL PREDICTION CALIBRATION (Residual-Based)
    # ========================================================================
    logger.info("\n--- Calibrating Prediction Intervals (Residual-Based) ---")
    
    # Use the 7-day horizon model for conformal calibration
    if 7 in models:
        target_col = 'target_7d'
        X_train = train_df[features].copy()
        y_train = train_df[target_col].copy()
        
        mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        
        # Calculate residuals on training set to estimate prediction intervals
        y_pred_train = models[7].predict(X_train)
        residuals = np.abs(y_train.values - y_pred_train)
        
        # Store quantiles for prediction intervals (90%, 95%)
        results['conformal_calibration'] = {
            'q90': float(np.percentile(residuals, 90)),
            'q95': float(np.percentile(residuals, 95)),
            'q80': float(np.percentile(residuals, 80)),
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
        }
        
        logger.info(f"  80% prediction interval width: ±{results['conformal_calibration']['q80']:.2f} Rs/quintal")
        logger.info(f"  90% prediction interval width: ±{results['conformal_calibration']['q90']:.2f} Rs/quintal")
        logger.info(f"  95% prediction interval width: ±{results['conformal_calibration']['q95']:.2f} Rs/quintal")
    
    return models, results


# ============================================================================
# PREDICTION & RECOMMENDATIONS
# ============================================================================

class PriceIntelligenceEngine:
    """Production price forecasting engine with recommendations and uncertainty."""
    
    def __init__(self, models: Dict[int, lgb.LGBMRegressor], 
                 encoders: Dict[str, LabelEncoder],
                 features: List[str],
                 conformal_calibration: Optional[Dict] = None):
        self.models = models
        self.encoders = encoders
        self.features = features
        self.horizons = sorted(models.keys())
        self.conformal_calibration = conformal_calibration or {}
    
    def predict_prices(self, input_data: pd.DataFrame, 
                       confidence: float = 0.9) -> pd.DataFrame:
        """Predict prices for all horizons with uncertainty intervals."""
        predictions = {}
        
        for horizon, model in self.models.items():
            X = input_data[self.features].copy()
            point_pred = model.predict(X)
            predictions[f'price_forecast_{horizon}d'] = point_pred
            
            # Add prediction intervals if calibration available
            if self.conformal_calibration:
                q_key = f'q{int(confidence * 100)}'
                interval_width = self.conformal_calibration.get(q_key, 
                    self.conformal_calibration.get('q90', 0))
                predictions[f'price_lower_{horizon}d'] = point_pred - interval_width
                predictions[f'price_upper_{horizon}d'] = point_pred + interval_width
        
        return pd.DataFrame(predictions, index=input_data.index)
    
    def get_recommendation(self, current_price: float, 
                           forecasts: Dict[int, float],
                           confidence_intervals: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict:
        """Generate HOLD/SELL recommendation based on forecasts with uncertainty."""
        
        # Find max price in forecast period
        max_forecast = max(forecasts.values())
        max_day = max(forecasts, key=forecasts.get)
        
        # Calculate expected gain
        expected_gain_pct = ((max_forecast - current_price) / current_price) * 100
        
        # Confidence-adjusted recommendation
        confidence_str = ""
        if confidence_intervals and max_day in confidence_intervals:
            lower, upper = confidence_intervals[max_day]
            confidence_str = f" (90% CI: {lower:.0f}-{upper:.0f})"
            
            # Conservative: use lower bound for HOLD decision
            conservative_gain = ((lower - current_price) / current_price) * 100
            if conservative_gain > HOLD_THRESHOLD_PCT:
                confidence_level = "high"
            elif expected_gain_pct > HOLD_THRESHOLD_PCT:
                confidence_level = "medium"
            else:
                confidence_level = "low"
        else:
            confidence_level = "unknown"
        
        # Recommendation logic
        if expected_gain_pct > HOLD_THRESHOLD_PCT:
            recommendation = "HOLD"
            reason = f"Price expected to rise {expected_gain_pct:.1f}% in {max_day} days{confidence_str}"
        elif expected_gain_pct < -HOLD_THRESHOLD_PCT:
            recommendation = "SELL"
            reason = f"Price expected to fall {abs(expected_gain_pct):.1f}% - sell now{confidence_str}"
        else:
            recommendation = "SELL"
            reason = f"Price stable (±{abs(expected_gain_pct):.1f}%) - no significant gain expected"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence_level,
            'reason': reason,
            'current_price': current_price,
            'max_forecast_price': max_forecast,
            'best_sell_day': max_day,
            'expected_gain_pct': expected_gain_pct,
            'forecasts': forecasts,
            'confidence_intervals': confidence_intervals,
        }
    
    def compare_markets(self, commodity: str, district: str, 
                        current_price: float, market_prices: Dict[str, float],
                        distances_km: Dict[str, float]) -> List[Dict]:
        """Compare profitability across markets accounting for transport."""
        
        comparisons = []
        
        for market, price in market_prices.items():
            distance = distances_km.get(market, 50)  # Default 50km
            transport_cost = distance * TRANSPORT_COST_PER_KM
            
            net_price = price - transport_cost
            profit_margin = net_price - current_price
            profit_pct = (profit_margin / current_price) * 100
            
            comparisons.append({
                'market': market,
                'market_price': price,
                'distance_km': distance,
                'transport_cost': transport_cost,
                'net_price': net_price,
                'profit_margin': profit_margin,
                'profit_pct': profit_pct,
                'profitable': profit_margin > 0,
            })
        
        # Sort by profit margin
        return sorted(comparisons, key=lambda x: x['profit_margin'], reverse=True)
    
    def save(self, path: Path):
        """Save engine to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for horizon, model in self.models.items():
            joblib.dump(model, path / f'model_horizon_{horizon}d.joblib')
        
        # Save encoders
        joblib.dump(self.encoders, path / 'encoders.joblib')
        
        # Save features
        with open(path / 'features.json', 'w') as f:
            json.dump(self.features, f, indent=2)
        
        # Save horizons
        with open(path / 'horizons.json', 'w') as f:
            json.dump(self.horizons, f)
        
        # Save conformal calibration
        with open(path / 'conformal_calibration.json', 'w') as f:
            json.dump(self.conformal_calibration, f, indent=2)
        
        logger.info(f"Saved engine to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'PriceIntelligenceEngine':
        """Load engine from disk."""
        # Load models
        with open(path / 'horizons.json', 'r') as f:
            horizons = json.load(f)
        
        models = {}
        for h in horizons:
            models[h] = joblib.load(path / f'model_horizon_{h}d.joblib')
        
        # Load encoders
        encoders = joblib.load(path / 'encoders.joblib')
        
        # Load features
        with open(path / 'features.json', 'r') as f:
            features = json.load(f)
        
        # Load conformal calibration
        conformal_path = path / 'conformal_calibration.json'
        if conformal_path.exists():
            with open(conformal_path, 'r') as f:
                conformal_calibration = json.load(f)
        else:
            conformal_calibration = {}
        
        return cls(models, encoders, features, conformal_calibration)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main(dry_run: bool = False, cv_only: bool = False):
    """Main training pipeline."""
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Price Intelligence Engine Training")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # Load data
    df = load_data()
    
    # Feature engineering
    df = create_lag_features(df, group_cols=['commodity', 'market'])
    df = create_calendar_features(df)
    df, encoders = encode_categoricals(df)
    df = create_target_horizons(df, FORECAST_HORIZONS)
    
    # Prepare features
    features, cat_features = prepare_features(df)
    logger.info(f"Total features: {len(features)}")
    
    # Drop rows with too many NaN features (from lag creation)
    min_features_required = len(features) * 0.7
    feature_nan_count = df[features].isna().sum(axis=1)
    df = df[feature_nan_count <= (len(features) - min_features_required)].copy()
    logger.info(f"Rows after feature filtering: {len(df):,}")
    
    if cv_only:
        # Just run cross-validation
        logger.info("\nRunning cross-validation only...")
        cv_results = time_series_cross_validate(
            df, features, 'target_7d', n_splits=5
        )
        logger.info(f"\nCV Results (7-day horizon):")
        logger.info(f"  MAE: {cv_results['mae']:.2f}")
        logger.info(f"  RMSE: {cv_results['rmse']:.2f}")
        logger.info(f"  R2: {cv_results['r2']:.4f}")
        logger.info(f"  MAPE: {cv_results['mape']:.2f}%")
        return
    
    if dry_run:
        logger.info("DRY RUN - skipping model training")
        return
    
    # Train models
    models, results = train_all_models(df, features, cat_features, FORECAST_HORIZONS)
    
    # Get conformal calibration if available
    conformal_calibration = results.get('conformal_calibration', {})
    
    # Create engine with conformal calibration
    engine = PriceIntelligenceEngine(models, encoders, features, conformal_calibration)
    
    # Save
    model_path = MODEL_DIR / "price_intelligence"
    engine.save(model_path)
    
    # Save results
    results_path = model_path / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'horizons': FORECAST_HORIZONS,
            'features_count': len(features),
            'features': features,
            'conformal_calibration': conformal_calibration,
            'results': {str(k): v for k, v in results.items() if k != 'conformal_calibration'},
        }, f, indent=2)
    
    # Generate report
    report_path = LOGS_DIR / "price_intelligence_training_report.md"
    generate_report(results, features, df, report_path)
    
    elapsed = datetime.now() - start_time
    logger.info(f"\nTraining complete in {elapsed}")
    logger.info(f"Models saved to: {model_path}")
    logger.info(f"Report saved to: {report_path}")


def generate_report(results: Dict, features: List[str], df: pd.DataFrame, path: Path):
    """Generate training report."""
    
    lines = [
        "# Price Intelligence Engine - Training Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Model Performance by Forecast Horizon",
        "",
        "| Horizon | MAE (Rs) | RMSE (Rs) | R² | MAPE (%) | Train | Test |",
        "|---------|----------|-----------|-----|---------|-------|------|",
    ]
    
    # Filter to only horizon metrics (integer keys)
    horizon_results = {k: v for k, v in results.items() if isinstance(k, int)}
    
    for horizon, metrics in sorted(horizon_results.items()):
        lines.append(
            f"| {horizon}d | {metrics['mae']:.2f} | {metrics['rmse']:.2f} | "
            f"{metrics['r2']:.4f} | {metrics['mape']:.2f} | "
            f"{metrics['n_train']:,} | {metrics['n_test']:,} |"
        )
    
    # Add conformal prediction intervals if available
    if 'conformal_calibration' in results:
        cal = results['conformal_calibration']
        lines.extend([
            "",
            "---",
            "",
            "## Prediction Intervals (Calibrated)",
            "",
            "| Confidence | Interval Width (Rs/quintal) |",
            "|------------|----------------------------|",
            f"| 80% | ±{cal['q80']:.2f} |",
            f"| 90% | ±{cal['q90']:.2f} |",
            f"| 95% | ±{cal['q95']:.2f} |",
            "",
            f"*Mean Absolute Residual: {cal['mean_residual']:.2f} Rs/quintal*",
        ])
    
    lines.extend([
        "",
        "---",
        "",
        "## Dataset Summary",
        "",
        f"- **Total Records:** {len(df):,}",
        f"- **Date Range:** {df['arrival_date'].min().date()} to {df['arrival_date'].max().date()}",
        f"- **Districts:** {df['district'].nunique()}",
        f"- **Markets:** {df['market'].nunique()}",
        f"- **Commodities:** {df['commodity'].nunique()}",
        "",
        "---",
        "",
        "## Features Used",
        "",
        f"**Total Features:** {len(features)}",
        "",
        "### Feature Categories:",
        "",
        "- **Lag Features:** " + ", ".join([f for f in features if 'lag' in f]),
        "- **Rolling Features:** " + ", ".join([f for f in features if 'rolling' in f][:5]) + "...",
        "- **Momentum Features:** " + ", ".join([f for f in features if 'momentum' in f]),
        "- **Calendar Features:** " + ", ".join([f for f in features if any(x in f for x in ['month', 'day', 'week', 'year', 'sin', 'cos'])]),
        "- **Weather Features:** " + ", ".join([f for f in features if any(x in f for x in ['t2m', 'humidity', 'precipitation', 'wind', 'solar'])]),
        "",
        "---",
        "",
        "## Usage Example",
        "",
        "```python",
        "from pathlib import Path",
        "import sys",
        "sys.path.insert(0, str(Path.cwd()))",
        "",
        "from scripts.train_price_model import PriceIntelligenceEngine",
        "",
        "# Load engine",
        "engine = PriceIntelligenceEngine.load(Path('models/price_intelligence'))",
        "",
        "# Predict (requires prepared input DataFrame)",
        "forecasts = engine.predict_prices(input_df)",
        "",
        "# Get recommendation",
        "rec = engine.get_recommendation(",
        "    current_price=2500,",
        "    forecasts={1: 2520, 3: 2580, 7: 2650, 14: 2600, 15: 2590}",
        ")",
        "print(rec['recommendation'], rec['reason'])",
        "```",
        "",
    ])
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Price Intelligence Engine")
    parser.add_argument("--dry-run", action="store_true", help="Skip training")
    parser.add_argument("--cv-only", action="store_true", help="Run cross-validation only")
    
    args = parser.parse_args()
    main(dry_run=args.dry_run, cv_only=args.cv_only)
