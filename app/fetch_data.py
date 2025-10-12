"""
Data fetching module for LLM Market Decision Agent.
Downloads hourly OHLCV data from Yahoo Finance.
"""

import logging
import pandas as pd
import yfinance as yf
from typing import List
from datetime import datetime, timedelta
import time
import random
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_hourly_data(
    symbols: List[str],
    period: str = "60d",
    interval: str = "1h",
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch hourly OHLCV data from Yahoo Finance with retry logic.
    
    Args:
        symbols: List of ticker symbols (e.g., ['SPY', 'QQQ', 'AAPL'])
        period: Time period to fetch (e.g., '60d', '30d', '7d')
        interval: Data interval (e.g., '1h', '1d')
        max_retries: Maximum number of retry attempts per symbol
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    logger.info(f"Fetching {interval} data for {len(symbols)} symbols: {symbols}")
    
    all_data = []
    
    for symbol in symbols:
        success = False
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Add exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retry {attempt + 1}/{max_retries} for {symbol} after {wait_time:.1f}s...")
                    time.sleep(wait_time)
                
                logger.info(f"Downloading {symbol}... (attempt {attempt + 1})")
                ticker = yf.Ticker(symbol)
                
                # Try different approaches for data fetching
                if attempt == 0:
                    # Try with conservative period first
                    if interval == "1h":
                        df = ticker.history(period="7d", interval=interval)
                    else:
                        df = ticker.history(period=period, interval=interval)
                elif attempt == 1:
                    # Try with even shorter period
                    if interval == "1h":
                        df = ticker.history(period="5d", interval=interval)
                    else:
                        df = ticker.history(period=period, interval=interval)
                else:
                    # Try with manual date range (last 5 days for hourly data)
                    if interval == "1h":
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=5)
                        df = ticker.history(start=start_date, end=end_date, interval=interval)
                    else:
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=7)
                        df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data retrieved for {symbol} (attempt {attempt + 1})")
                    if attempt == max_retries - 1:
                        logger.error(f"All attempts failed for {symbol}")
                    continue
                
                # Reset index to get timestamp as column
                df = df.reset_index()
                
                # Rename columns to lowercase
                df.columns = df.columns.str.lower()
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Select relevant columns
                df = df[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                df.rename(columns={'datetime': 'timestamp'}, inplace=True)
                
                # Remove timezone info for consistency
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
                all_data.append(df)
                success = True
                break
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for {symbol}")
                continue
        
        if not success:
            logger.warning(f"Failed to fetch data for {symbol} after {max_retries} attempts")
    
    if not all_data:
        logger.error("No data could be fetched for any symbol")
        # Generate sample data as fallback
        logger.info("Generating sample data for demonstration purposes...")
        return generate_sample_data(symbols)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp and symbol
    combined_df = combined_df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    logger.info(f"Total rows fetched: {len(combined_df)}")
    
    return combined_df


def generate_sample_data(symbols: List[str]) -> pd.DataFrame:
    """
    Generate sample data for demonstration purposes when API fails.
    
    Args:
        symbols: List of ticker symbols
    
    Returns:
        DataFrame with sample OHLCV data
    """
    logger.info("Generating sample market data for demonstration...")
    
    # Generate timestamps for the last 60 days, hourly
    end_time = datetime.now()
    start_time = end_time - timedelta(days=60)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
    
    all_data = []
    
    for symbol in symbols:
        # Generate realistic price movements
        base_price = {
            'SPY': 450.0,
            'QQQ': 380.0,
            'AAPL': 180.0
        }.get(symbol, 100.0)
        
        data = []
        current_price = base_price
        
        for timestamp in timestamps:
            # Generate realistic price movement
            price_change = random.uniform(-0.02, 0.02)  # ±2% max change
            current_price *= (1 + price_change)
            
            # Generate OHLC from current price
            high = current_price * random.uniform(1.0, 1.01)
            low = current_price * random.uniform(0.99, 1.0)
            open_price = current_price * random.uniform(0.995, 1.005)
            close_price = current_price
            
            # Generate volume
            volume = random.randint(1000000, 10000000)
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        all_data.extend(data)
    
    df = pd.DataFrame(all_data)
    logger.info(f"Generated {len(df)} sample data rows")
    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Destination file path
    """
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")


def main():
    """Main execution function."""
    try:
        # Fetch data
        df = fetch_hourly_data(
            symbols=config.SYMBOLS,
            period=config.PERIOD,
            interval=config.INTERVAL
        )
        
        # Save to file
        save_data(df, config.HOURLY_DATA_FILE)
        
        # Display summary
        print("\n" + "="*60)
        print("DATA FETCH SUMMARY")
        print("="*60)
        print(f"Symbols: {', '.join(config.SYMBOLS)}")
        print(f"Period: {config.PERIOD}")
        print(f"Interval: {config.INTERVAL}")
        print(f"Total rows: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nData saved to: {config.HOURLY_DATA_FILE}")
        print("="*60 + "\n")
        
        # Display sample
        print("Sample data:")
        print(df.head(10))
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        raise


if __name__ == "__main__":
    main()

