"""
Multi-timeframe data fetching module.
Fetches market data across different timeframes (1h, 4h, 1d, 1w) for comparative analysis.
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTimeframeDataFetcher:
    """Fetches market data across multiple timeframes for comparative analysis."""
    
    def __init__(self):
        """Initialize the multi-timeframe data fetcher."""
        self.symbols = config.SYMBOLS
        self.timeframes = config.TIMEFRAMES
        logger.info(f"Initialized fetcher for symbols: {self.symbols}")
        logger.info(f"Available timeframes: {list(self.timeframes.keys())}")
    
    def fetch_timeframe_data(self, timeframe: str) -> pd.DataFrame:
        """
        Fetch data for a specific timeframe.
        
        Args:
            timeframe: Timeframe key (1h, 4h, 1d, 1w)
            
        Returns:
            DataFrame with OHLCV data for the timeframe
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        tf_config = self.timeframes[timeframe]
        interval = tf_config['interval']
        period = tf_config['period']
        suffix = tf_config['file_suffix']
        
        logger.info(f"Fetching {timeframe} data (interval: {interval}, period: {period})")
        
        all_data = []
        
        for symbol in self.symbols:
            try:
                logger.info(f"Downloading {symbol}... (attempt 1)")
                
                # Download data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    logger.warning(f"No data retrieved for {symbol} ({timeframe})")
                    continue
                
                # Reset index and clean data
                data = data.reset_index()
                data['symbol'] = symbol
                data['timeframe'] = timeframe
                
                # Rename columns to match our standard format
                data = data.rename(columns={
                    'Datetime': 'timestamp',
                    'Date': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Ensure timestamp is datetime
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                else:
                    logger.warning(f"No timestamp column found for {symbol}")
                    continue
                
                all_data.append(data)
                logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} ({timeframe}): {e}")
                continue
        
        if not all_data:
            logger.error("No data retrieved for any symbols")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        combined_data = combined_data.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"Total {timeframe} rows fetched: {len(combined_data)}")
        
        return combined_data
    
    def fetch_all_timeframes(self, skip_1h: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all configured timeframes.
        
        Args:
            skip_1h: If True, skip 1h timeframe (already fetched by main pipeline)
        
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        logger.info("Fetching data for all timeframes...")
        
        all_timeframes = {}
        timeframes_to_fetch = list(self.timeframes.keys())
        
        if skip_1h and '1h' in timeframes_to_fetch:
            timeframes_to_fetch.remove('1h')
            logger.info("Skipping 1h timeframe (already fetched by main pipeline)")
        
        for timeframe in timeframes_to_fetch:
            try:
                data = self.fetch_timeframe_data(timeframe)
                if not data.empty:
                    all_timeframes[timeframe] = data
                    logger.info(f"✓ {timeframe} data fetched: {len(data)} rows")
                else:
                    logger.warning(f"✗ No data for {timeframe}")
            except Exception as e:
                logger.error(f"✗ Error fetching {timeframe}: {e}")
        
        return all_timeframes
    
    def save_timeframe_data(self, timeframe_data: Dict[str, pd.DataFrame]):
        """
        Save timeframe data to separate CSV files.
        
        Args:
            timeframe_data: Dictionary of timeframe DataFrames
        """
        logger.info("Saving timeframe data to files...")
        
        for timeframe, df in timeframe_data.items():
            if df.empty:
                continue
                
            tf_config = self.timeframes[timeframe]
            suffix = tf_config['file_suffix']
            
            filename = config.DATA_DIR / f"{suffix}_data.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved {timeframe} data to {filename}")
    
    def create_comparative_analysis(self, timeframe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create comparative analysis across timeframes.
        
        Args:
            timeframe_data: Dictionary of timeframe DataFrames
            
        Returns:
            DataFrame with comparative metrics
        """
        logger.info("Creating comparative analysis across timeframes...")
        
        comparison_data = []
        
        for timeframe, df in timeframe_data.items():
            if df.empty:
                continue
            
            # Calculate timeframe-specific metrics
            tf_metrics = self._calculate_timeframe_metrics(df, timeframe)
            comparison_data.append(tf_metrics)
        
        if not comparison_data:
            logger.warning("No data available for comparison")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_file = config.DATA_DIR / "timeframe_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Saved timeframe comparison to {comparison_file}")
        
        return comparison_df
    
    def _calculate_timeframe_metrics(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Calculate metrics for a specific timeframe."""
        metrics = {
            'timeframe': timeframe,
            'total_rows': len(df),
            'symbols': df['symbol'].nunique(),
            'date_range_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'avg_volume': df['volume'].mean(),
            'price_volatility': df.groupby('symbol')['close'].std().mean(),
        }
        
        # Calculate symbol-specific metrics
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            if len(symbol_data) > 1:
                price_change = ((symbol_data['close'].iloc[-1] - symbol_data['close'].iloc[0]) / 
                               symbol_data['close'].iloc[0]) * 100
                metrics[f'{symbol}_price_change_pct'] = price_change
                metrics[f'{symbol}_volume_trend'] = symbol_data['volume'].mean()
        
        return metrics
    
    def generate_timeframe_report(self, timeframe_data: Dict[str, pd.DataFrame]) -> str:
        """Generate a comprehensive report comparing timeframes."""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("MULTI-TIMEFRAME DATA ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("## Data Summary")
        report_lines.append("")
        
        for timeframe, df in timeframe_data.items():
            if df.empty:
                continue
                
            tf_config = self.timeframes[timeframe]
            report_lines.append(f"### {timeframe.upper()} Timeframe")
            report_lines.append(f"- Interval: {tf_config['interval']}")
            report_lines.append(f"- Period: {tf_config['period']}")
            report_lines.append(f"- Total Rows: {len(df)}")
            report_lines.append(f"- Symbols: {', '.join(df['symbol'].unique())}")
            report_lines.append(f"- Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Price statistics
            price_stats = df.groupby('symbol')['close'].agg(['mean', 'std', 'min', 'max'])
            report_lines.append("- Price Statistics:")
            for symbol in df['symbol'].unique():
                stats = price_stats.loc[symbol]
                report_lines.append(f"  * {symbol}: Mean=${stats['mean']:.2f}, Std=${stats['std']:.2f}")
            
            report_lines.append("")
        
        # Comparative analysis
        report_lines.append("## Comparative Analysis")
        report_lines.append("")
        
        comparison_df = self.create_comparative_analysis(timeframe_data)
        
        if not comparison_df.empty:
            report_lines.append("### Timeframe Comparison Table")
            report_lines.append("")
            report_lines.append(comparison_df.to_string(index=False))
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("1. **1h timeframe**: Best for intraday trading and short-term analysis")
        report_lines.append("2. **4h timeframe**: Good balance for swing trading strategies")
        report_lines.append("3. **1d timeframe**: Ideal for position trading and trend analysis")
        report_lines.append("4. **1w timeframe**: Best for long-term investment decisions")
        report_lines.append("")
        report_lines.append("Consider running LLM analysis on multiple timeframes to get")
        report_lines.append("comprehensive market insights across different trading horizons.")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = config.DATA_DIR / "timeframe_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Timeframe analysis report saved to {report_file}")
        
        return report_text


def main():
    """Test the multi-timeframe data fetcher."""
    logger.info("Testing Multi-Timeframe Data Fetcher...")
    
    # Initialize fetcher
    fetcher = MultiTimeframeDataFetcher()
    
    # Fetch data for all timeframes (skip 1h as it's already fetched)
    timeframe_data = fetcher.fetch_all_timeframes(skip_1h=True)
    
    if not timeframe_data:
        logger.error("No data fetched for any timeframe")
        return
    
    # Save data
    fetcher.save_timeframe_data(timeframe_data)
    
    # Generate report
    report = fetcher.generate_timeframe_report(timeframe_data)
    
    # Print summary
    print("\n" + "="*70)
    print("MULTI-TIMEFRAME FETCH SUMMARY")
    print("="*70)
    
    for timeframe, df in timeframe_data.items():
        print(f"\n{timeframe.upper()} Timeframe:")
        print(f"  Rows: {len(df)}")
        print(f"  Symbols: {', '.join(df['symbol'].unique())}")
        print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print(f"\n✅ Multi-timeframe data fetch completed.")
    print(f"📊 Generated analysis report: data/timeframe_analysis_report.txt")
    print(f"📈 Saved comparison data: data/timeframe_comparison.csv")


if __name__ == "__main__":
    main()
