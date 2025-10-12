"""
Feature computation module for LLM Market Decision Agent.
Computes technical indicators and Weighted Sentiment Score (WSS).
"""

import logging
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from typing import Optional
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period (default: 14)
    
    Returns:
        Series with RSI values
    """
    rsi_indicator = RSIIndicator(close=df['close'], window=period)
    return rsi_indicator.rsi()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default: 14)
    
    Returns:
        Series with ATR values
    """
    atr_indicator = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=period
    )
    return atr_indicator.average_true_range()


def compute_volume_bias(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute Volume Bias (current volume / MA of volume).
    
    Args:
        df: DataFrame with 'volume' column
        period: Moving average period (default: 20)
    
    Returns:
        Series with volume bias values
    """
    volume_ma = df['volume'].rolling(window=period).mean()
    return df['volume'] / volume_ma


def compute_trend(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Compute trend direction based on SMA slope.
    
    Args:
        df: DataFrame with 'close' column
        period: SMA period (default: 10)
    
    Returns:
        Series with trend values ('up', 'down', 'neutral')
    """
    sma = df['close'].rolling(window=period).mean()
    slope = sma.diff(periods=1)
    
    # Classify trend based on slope
    trend = pd.Series('neutral', index=df.index)
    trend[slope > 0] = 'up'
    trend[slope < 0] = 'down'
    
    return trend


def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize series to 0-1 range using min-max scaling.
    
    Args:
        series: Input series
    
    Returns:
        Normalized series
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    
    return (series - min_val) / (max_val - min_val)


def compute_wss(
    rsi: pd.Series,
    atr: pd.Series,
    volume_bias: pd.Series,
    rsi_weight: float = 0.4,
    volume_weight: float = 0.4,
    atr_weight: float = 0.2
) -> pd.Series:
    """
    Compute Weighted Sentiment Score (WSS).
    
    WSS combines normalized indicators:
    - RSI (normalized to 0-1)
    - Volume Bias (normalized to 0-1)
    - Inverse ATR (normalized, higher ATR = lower score = more risk)
    
    Args:
        rsi: RSI series
        atr: ATR series
        volume_bias: Volume bias series
        rsi_weight: Weight for RSI (default: 0.4)
        volume_weight: Weight for volume bias (default: 0.4)
        atr_weight: Weight for inverse ATR (default: 0.2)
    
    Returns:
        Series with WSS values (0-1 range)
    """
    # Normalize each component
    rsi_norm = rsi / 100  # RSI is already 0-100
    volume_norm = normalize_series(volume_bias.clip(lower=0, upper=3))  # Clip outliers
    atr_norm = normalize_series(atr)
    inverse_atr_norm = 1 - atr_norm  # Inverse: lower volatility = higher score
    
    # Weighted combination
    wss = (
        rsi_weight * rsi_norm +
        volume_weight * volume_norm +
        atr_weight * inverse_atr_norm
    )
    
    return wss.clip(lower=0, upper=1)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and features for each symbol.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added feature columns
    """
    logger.info("Computing technical indicators...")
    
    # Process each symbol separately
    symbol_dfs = []
    
    for symbol in df['symbol'].unique():
        logger.info(f"Processing {symbol}...")
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Compute indicators
        symbol_df['rsi'] = compute_rsi(symbol_df, period=config.RSI_PERIOD)
        symbol_df['atr'] = compute_atr(symbol_df, period=config.ATR_PERIOD)
        symbol_df['volume_bias'] = compute_volume_bias(symbol_df, period=config.VOLUME_MA_PERIOD)
        symbol_df['trend'] = compute_trend(symbol_df, period=config.TREND_SMA_PERIOD)
        
        # Compute WSS
        symbol_df['wss'] = compute_wss(
            rsi=symbol_df['rsi'],
            atr=symbol_df['atr'],
            volume_bias=symbol_df['volume_bias']
        )
        
        symbol_dfs.append(symbol_df)
    
    # Combine all symbols
    result_df = pd.concat(symbol_dfs, ignore_index=True)
    
    # Drop rows with NaN values (from indicator warmup period)
    initial_rows = len(result_df)
    result_df = result_df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(result_df)
    
    logger.info(f"Dropped {dropped_rows} rows with NaN values (indicator warmup)")
    logger.info(f"Final dataset: {len(result_df)} rows")
    
    return result_df


def load_hourly_data(filepath: str) -> pd.DataFrame:
    """
    Load hourly data from CSV file.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame with hourly data
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def save_features(df: pd.DataFrame, filepath: str) -> None:
    """
    Save features DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Destination file path
    """
    df.to_csv(filepath, index=False)
    logger.info(f"Features saved to {filepath}")


def main():
    """Main execution function."""
    try:
        # Load hourly data
        logger.info(f"Loading data from {config.HOURLY_DATA_FILE}")
        df = load_hourly_data(config.HOURLY_DATA_FILE)
        
        # Compute features
        features_df = compute_features(df)
        
        # Save features
        save_features(features_df, config.FEATURES_FILE)
        
        # Display summary
        print("\n" + "="*60)
        print("FEATURE COMPUTATION SUMMARY")
        print("="*60)
        print(f"Total rows processed: {len(features_df)}")
        print(f"Symbols: {', '.join(features_df['symbol'].unique())}")
        print(f"\nFeatures computed:")
        print(f"  - RSI(14)")
        print(f"  - ATR(14)")
        print(f"  - Volume Bias (vs 20-hour MA)")
        print(f"  - Trend (10-hour SMA slope)")
        print(f"  - Weighted Sentiment Score (WSS)")
        print(f"\nFeatures saved to: {config.FEATURES_FILE}")
        print("="*60 + "\n")
        
        # Display sample
        print("Sample features:")
        print(features_df[['timestamp', 'symbol', 'close', 'rsi', 'atr', 
                          'volume_bias', 'trend', 'wss']].head(10))
        
        # Display statistics
        print("\nFeature Statistics:")
        print(features_df[['rsi', 'atr', 'volume_bias', 'wss']].describe())
        
    except Exception as e:
        logger.error(f"Failed to compute features: {str(e)}")
        raise


if __name__ == "__main__":
    main()

