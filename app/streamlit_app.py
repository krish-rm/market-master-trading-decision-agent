"""
Streamlit frontend for LLM Market Decision Agent.
Interactive dashboard displaying market data, indicators, and LLM insights.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import config
except ImportError:
    # Fallback for when running from different directory
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LLM_OUTPUTS_FILE = DATA_DIR / "llm_outputs.csv"
    FEATURES_FILE = DATA_DIR / "features.csv"


# Page configuration
st.set_page_config(
    page_title="LLM Market Decision Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    """Load all available data including multi-timeframe data."""
    try:
        # Try to use config
        data_dir = Path(config.DATA_DIR)
    except:
        # Fallback
        data_dir = Path(__file__).parent.parent / "data"
    
    data_files = {
        'hourly_data': data_dir / "hourly_data.csv",
        'features': data_dir / "features.csv", 
        'llm_outputs': data_dir / "llm_outputs.csv",
        '4hourly_data': data_dir / "4hourly_data.csv",
        'daily_data': data_dir / "daily_data.csv",
        'weekly_data': data_dir / "weekly_data.csv",
        'timeframe_comparison': data_dir / "timeframe_comparison.csv",
        'news_data': data_dir / "news_data.json"
    }
    
    loaded_data = {}
    
    # Load all available CSV files
    for key, file_path in data_files.items():
        if file_path.suffix == '.csv' and file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if 'timestamp' in df.columns:
                    # Handle timezone-aware timestamps properly
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    # Convert to timezone-naive for compatibility
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                loaded_data[key] = df
            except Exception as e:
                st.warning(f"Could not load {key}: {e}")
    
    # Load news data if available
    if data_files['news_data'].exists():
        try:
            import json
            with open(data_files['news_data'], 'r') as f:
                loaded_data['news_data'] = json.load(f)
        except Exception as e:
            st.warning(f"Could not load news data: {e}")
    
    return loaded_data


def create_price_chart(df, symbol):
    """Create interactive price and indicator chart."""
    symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')
    
    # Check what indicators are available
    has_rsi = 'rsi' in symbol_df.columns
    has_wss = 'wss' in symbol_df.columns
    has_volume_bias = 'volume_bias' in symbol_df.columns
    
    # Determine number of rows based on available indicators
    rows = 1
    if has_rsi or has_wss:
        rows += 1
    if has_volume_bias:
        rows += 1
    
    if rows == 1:
        # Simple price chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=symbol_df['timestamp'],
                y=symbol_df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#2E86DE', width=2)
            )
        )
        
        # Add candlestick if we have OHLC data
        if all(col in symbol_df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=symbol_df['timestamp'],
                    open=symbol_df['open'],
                    high=symbol_df['high'],
                    low=symbol_df['low'],
                    close=symbol_df['close'],
                    name='OHLC',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                )
            )
        
        fig.update_layout(
            height=500,
            title_text=f"{symbol} Price Chart",
            title_font_size=20,
            xaxis_title="Time",
            yaxis_title="Price ($)"
        )
        
    else:
        # Multi-row chart with indicators
        subplot_titles = ['Price & Volume']
        row_heights = [0.7]
        
        if has_rsi or has_wss:
            subplot_titles.append('Technical Indicators')
            row_heights.append(0.3)
        if has_volume_bias:
            subplot_titles.append('Volume Analysis')
            row_heights.append(0.3)
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=symbol_df['timestamp'],
                y=symbol_df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#2E86DE', width=2)
            ),
            row=1, col=1
        )
        
        # Volume if available
        if 'volume' in symbol_df.columns:
            fig.add_trace(
                go.Bar(
                    x=symbol_df['timestamp'],
                    y=symbol_df['volume'],
                    name='Volume',
                    marker_color='#95A5A6',
                    opacity=0.3,
                    yaxis='y2'
                ),
                row=1, col=1
            )
        
        # Technical indicators
        if has_rsi or has_wss:
            row_idx = 2
            
            if has_rsi:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_df['timestamp'],
                        y=symbol_df['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#FF6B6B', width=2)
                    ),
                    row=row_idx, col=1
                )
                
                # RSI overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=row_idx, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=row_idx, col=1)
            
            if has_wss:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_df['timestamp'],
                        y=symbol_df['wss'],
                        mode='lines',
                        name='WSS',
                        line=dict(color='#4ECDC4', width=2)
                    ),
                    row=row_idx, col=1
                )
            
            row_idx += 1
        
        # Volume bias
        if has_volume_bias:
            fig.add_trace(
                go.Bar(
                    x=symbol_df['timestamp'],
                    y=symbol_df['volume_bias'],
                    name='Volume Bias',
                    marker_color='#F39C12'
                ),
                row=row_idx, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            title_text=f"{symbol} Market Analysis",
            title_font_size=20
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=rows, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if has_rsi or has_wss:
            fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
        if has_volume_bias:
            fig.update_yaxes(title_text="Volume Bias", row=rows, col=1)
    
    return fig


def display_llm_insight(row):
    """Display LLM-generated insight in a formatted card."""
    # Determine confidence color
    confidence_colors = {
        'High': '#27AE60',
        'Medium': '#F39C12',
        'Low': '#E74C3C'
    }
    confidence_color = confidence_colors.get(row['confidence'], '#95A5A6')
    
    # Create styled card
    st.markdown(f"""
    <div style="
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {confidence_color};
        margin: 10px 0;
    ">
        <h3 style="margin-top: 0; color: #2C3E50;">
            {row['symbol']} - {row['timestamp'].strftime('%Y-%m-%d %H:%M')}
        </h3>
        <p style="color: #7F8C8D; margin: 5px 0;">
            <strong>Price:</strong> ${row['close']:.2f} | 
            <strong>RSI:</strong> {row['rsi']:.1f} | 
            <strong>WSS:</strong> {row['wss']:.2f} | 
            <strong>Trend:</strong> {row['trend'].upper()}
        </p>
        <hr style="border: 1px solid #E0E0E0;">
        <h4 style="color: #34495E;">💡 Reasoning</h4>
        <p style="color: #2C3E50; line-height: 1.6;">{row['reasoning']}</p>
        <h4 style="color: #34495E;">📊 Guidance</h4>
        <p style="color: #2C3E50; line-height: 1.6;">{row['guidance']}</p>
        <div style="
            background-color: {confidence_color};
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-top: 15px;
            text-align: center;
            font-weight: bold;
        ">
            Confidence: {row['confidence']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📈 LLM Market Decision Agent - Advanced")
    st.markdown("""
    AI-powered market analysis with **RAG**, **Multi-timeframe Analysis**, and **Advanced Evaluation**
    using **Groq** to interpret technical indicators and provide adaptive trading guidance.
    """)
    
    # Load data
    with st.spinner("Loading multi-timeframe data..."):
        data = load_data()
    
    if not data:
        st.error("""
        ⚠️ No data found. Please run the advanced pipeline first:
        `python run_pipeline_advanced.py`
        """)
        return
    
    # Check what data we have
    available_data = list(data.keys())
    st.success(f"✅ Loaded data: {', '.join(available_data)}")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Get all symbols from available data
    all_symbols = set()
    for key, df in data.items():
        if isinstance(df, pd.DataFrame) and 'symbol' in df.columns:
            all_symbols.update(df['symbol'].unique())
    
    symbols = sorted(list(all_symbols))
    selected_symbol = st.sidebar.selectbox(
        "Select Symbol",
        symbols,
        index=0
    )
    
    # Timeframe selector
    available_timeframes = []
    if 'hourly_data' in data:
        available_timeframes.append('1h')
    if '4hourly_data' in data:
        available_timeframes.append('4h')
    if 'daily_data' in data:
        available_timeframes.append('1d')
    if 'weekly_data' in data:
        available_timeframes.append('1w')
    
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        available_timeframes,
        index=0
    )
    
    # Get data for selected timeframe
    timeframe_map = {
        '1h': 'hourly_data',
        '4h': '4hourly_data', 
        '1d': 'daily_data',
        '1w': 'weekly_data'
    }
    
    current_data = data.get(timeframe_map[selected_timeframe])
    if current_data is None:
        st.error(f"No data available for {selected_timeframe} timeframe")
        return
    
    # Ensure timestamp is datetime
    if 'timestamp' in current_data.columns:
        current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
    
    # Date range
    min_date = current_data['timestamp'].min().date()
    max_date = current_data['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data
    filtered_df = current_data[
        (current_data['symbol'] == selected_symbol) &
        (current_data['timestamp'].dt.date >= date_range[0])
    ]
    
    if len(date_range) > 1:
        filtered_df = filtered_df[
            filtered_df['timestamp'].dt.date <= date_range[1]
        ]
    
    # Sort by timestamp descending for latest first
    filtered_df = filtered_df.sort_values('timestamp', ascending=False)
    
    # Sidebar statistics
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Statistics")
    st.sidebar.metric("Total Data Points", len(filtered_df))
    st.sidebar.metric("Timeframe", selected_timeframe)
    
    # Show LLM insights count if available
    if 'llm_outputs' in data:
        llm_df = data['llm_outputs']
        llm_filtered = llm_df[llm_df['symbol'] == selected_symbol]
        st.sidebar.metric("LLM Insights", len(llm_filtered))
    
    # Main content
    if len(filtered_df) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Multi-Timeframe Charts", "🤖 LLM Insights", "📋 Data Tables", "📰 News Context"])
    
    with tab1:
        st.subheader(f"{selected_symbol} Multi-Timeframe Analysis")
        
        # Create a comprehensive price chart for the selected timeframe
        if 'close' in filtered_df.columns:
            chart = create_price_chart(filtered_df, selected_symbol)
            st.plotly_chart(chart, use_container_width=True)
        
        # Show key metrics if available
        if len(filtered_df) > 0:
            latest = filtered_df.iloc[0]
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Latest Price", f"${latest['close']:.2f}")
            
            # Show available metrics
            if 'rsi' in latest:
                col2.metric("RSI", f"{latest['rsi']:.1f}")
            if 'wss' in latest:
                col3.metric("WSS", f"{latest['wss']:.2f}")
            if 'volume_bias' in latest:
                col4.metric("Volume Bias", f"{latest['volume_bias']:.2f}x")
            if 'trend' in latest:
                col5.metric("Trend", latest['trend'].upper())
        
        # Show timeframe comparison if available
        if 'timeframe_comparison' in data:
            st.subheader("📊 Timeframe Comparison")
            comparison_df = data['timeframe_comparison']
            
            # Create comparison chart using actual data from each timeframe
            fig = go.Figure()
            
            for timeframe in ['1h', '4h', '1d', '1w']:
                timeframe_map_key = {
                    '1h': 'hourly_data',
                    '4h': '4hourly_data', 
                    '1d': 'daily_data',
                    '1w': 'weekly_data'
                }.get(timeframe)
                
                if timeframe_map_key and timeframe_map_key in data:
                    tf_data = data[timeframe_map_key]
                    symbol_data = tf_data[tf_data['symbol'] == selected_symbol].sort_values('timestamp')
                    
                    if len(symbol_data) > 0:
                        fig.add_trace(go.Scatter(
                            x=symbol_data['timestamp'],
                            y=symbol_data['close'],
                            mode='lines',
                            name=f'{timeframe} ({len(symbol_data)} points)',
                            line=dict(width=2)
                        ))
            
            if len(fig.data) > 1:  # Only show if we have multiple timeframes
                fig.update_layout(
                    title=f"{selected_symbol} Price Comparison Across Timeframes",
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show comparison statistics
            st.write("**Timeframe Statistics:**")
            display_cols = ['timeframe', 'total_rows', 'symbols', 'date_range_days', 'price_volatility']
            if f'{selected_symbol}_price_change_pct' in comparison_df.columns:
                display_cols.append(f'{selected_symbol}_price_change_pct')
            if f'{selected_symbol}_volume_trend' in comparison_df.columns:
                display_cols.append(f'{selected_symbol}_volume_trend')
            
            st.dataframe(comparison_df[display_cols], use_container_width=True)
    
    with tab2:
        st.subheader("🤖 LLM-Generated Insights")
        
        # Check if LLM insights are available for the selected timeframe
        if selected_timeframe != '1h':
            st.info(f"""
            **⚠️ LLM Insights Only Available for 1-Hour Data**
            
            LLM analysis was performed only on 1-hour timeframe data. 
            Switch to **1h timeframe** to see AI-generated insights for {selected_symbol}.
            
            **Available Timeframes with LLM Insights:**
            - ✅ **1h**: AI-powered analysis available
            - ❌ **4h**: No LLM analysis (raw data only)
            - ❌ **1d**: No LLM analysis (raw data only)  
            - ❌ **1w**: No LLM analysis (raw data only)
            """)
            
            # Show a preview of what 1h LLM insights look like
            if 'llm_outputs' in data:
                llm_df = data['llm_outputs']
                symbol_llm = llm_df[llm_df['symbol'] == selected_symbol]
                
                if len(symbol_llm) > 0:
                    st.write("**Preview of 1-hour LLM Insights:**")
                    preview_row = symbol_llm.iloc[0]
                    st.write(f"📊 **Sample Insight**: {preview_row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"💡 **Reasoning**: {preview_row['reasoning'][:150]}...")
                    st.write(f"🎯 **Guidance**: {preview_row['guidance']}")
                    st.write(f"📈 **Confidence**: {preview_row['confidence']}")
                    
                    st.write(f"\n**Total 1-hour insights available**: {len(symbol_llm)} for {selected_symbol}")
        else:
            # Show LLM insights for 1h timeframe
            if 'llm_outputs' in data:
                llm_df = data['llm_outputs']
                symbol_llm = llm_df[llm_df['symbol'] == selected_symbol]
                
                if len(symbol_llm) > 0:
                    st.success(f"✅ **LLM Analysis Available for {selected_symbol} (1h timeframe)**")
                    num_insights = st.slider("Number of insights to display", 1, 20, min(5, len(symbol_llm)))
                    
                    for _, row in symbol_llm.head(num_insights).iterrows():
                        display_llm_insight(row)
                else:
                    # Show what data is available for this symbol
                    st.info(f"**No LLM insights available for {selected_symbol}**")
                    
                    # Check if we have features data for this symbol
                    if 'features' in data:
                        features_df = data['features']
                        symbol_features = features_df[features_df['symbol'] == selected_symbol]
                        
                        if len(symbol_features) > 0:
                            st.write(f"✅ **Available data for {selected_symbol}:**")
                            st.write(f"- **Features data**: {len(symbol_features)} data points")
                            st.write(f"- **Date range**: {symbol_features['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {symbol_features['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"- **Technical indicators**: RSI, ATR, Volume Bias, Trend, WSS")
                            
                            # Show sample features
                            st.write("**Sample technical indicators:**")
                            sample_features = symbol_features[['timestamp', 'close', 'rsi', 'wss', 'trend']].head(3)
                            st.dataframe(sample_features, use_container_width=True)
                            
                            st.write("💡 **Note**: LLM analysis was only performed on 1-hour timeframe data. Switch to 1h timeframe to see AI insights.")
                        else:
                            st.warning(f"No features data available for {selected_symbol}")
            else:
                st.info("No LLM insights data available")
        
        # Show overall LLM insights summary
        if 'llm_outputs' in data:
            llm_df = data['llm_outputs']
            st.write("---")
            st.write("**LLM Insights Summary:**")
            
            # Show insights by symbol
            insights_by_symbol = llm_df['symbol'].value_counts()
            col1, col2, col3 = st.columns(3)
            
            for i, (symbol, count) in enumerate(insights_by_symbol.items()):
                if i == 0:
                    col1.metric(f"{symbol} Insights", count)
                elif i == 1:
                    col2.metric(f"{symbol} Insights", count)
                else:
                    col3.metric(f"{symbol} Insights", count)
            
            # Show date range of insights
            if len(llm_df) > 0:
                st.write(f"**Insights Date Range**: {llm_df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {llm_df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
                
                # Show confidence distribution
                confidence_dist = llm_df['confidence'].value_counts()
                st.write("**Confidence Distribution:**")
                for conf, count in confidence_dist.items():
                    st.write(f"- {conf}: {count} insights")
    
    with tab3:
        st.subheader("📋 Data Tables")
        
        # Show current timeframe data
        st.write(f"**{selected_timeframe} Data for {selected_symbol}**")
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Show all timeframes summary
        if len(available_timeframes) > 1:
            st.write("**All Timeframes Summary**")
            summary_data = []
            for tf in available_timeframes:
                tf_data = data.get(timeframe_map[tf])
                if tf_data is not None:
                    symbol_data = tf_data[tf_data['symbol'] == selected_symbol]
                    if len(symbol_data) > 0:
                        summary_data.append({
                            'Timeframe': tf,
                            'Data Points': len(symbol_data),
                            'Date Range': f"{symbol_data['timestamp'].min().strftime('%Y-%m-%d')} to {symbol_data['timestamp'].max().strftime('%Y-%m-%d')}",
                            'Latest Price': f"${symbol_data['close'].iloc[-1]:.2f}"
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
    
    with tab4:
        st.subheader("📰 News Context (RAG)")
        
        if 'news_data' in data and data['news_data']:
            news_data = data['news_data']
            
            # Handle different news data structures
            if isinstance(news_data, dict):
                if 'news_articles' in news_data:
                    articles = news_data['news_articles']
                else:
                    articles = [news_data]  # Single article in dict format
            elif isinstance(news_data, list):
                articles = news_data
            else:
                articles = []
            
            st.write(f"**Available News Articles: {len(articles)}**")
            
            # Show news articles (first 5)
            for i, article in enumerate(articles[:5]):
                # Handle different field names
                title = article.get('title', 'No Title')
                source = article.get('source', 'Unknown')
                published = article.get('publishedAt') or article.get('published_at', 'Unknown')
                description = article.get('description', 'No description available')
                similarity_score = article.get('similarity_score', 0)
                symbol = article.get('symbol', 'General')
                
                # Color code by similarity score
                border_color = '#E74C3C' if similarity_score < 0.5 else '#F39C12' if similarity_score < 0.8 else '#27AE60'
                
                st.markdown(f"""
                <div style="
                    background-color: #F8F9FA;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid {border_color};
                    margin: 10px 0;
                ">
                    <h4 style="margin-top: 0; color: #2C3E50;">{title}</h4>
                    <p style="color: #7F8C8D; margin: 5px 0;">
                        <strong>Symbol:</strong> {symbol} | 
                        <strong>Source:</strong> {source} | 
                        <strong>Published:</strong> {published} |
                        <strong>Relevance:</strong> {similarity_score:.2f}
                    </p>
                    <p style="color: #2C3E50; line-height: 1.6;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show news summary statistics
            if articles:
                st.write("**News Summary:**")
                col1, col2, col3 = st.columns(3)
                
                # Count by symbol
                symbol_counts = {}
                for article in articles:
                    symbol = article.get('symbol', 'General')
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                
                col1.metric("Total Articles", len(articles))
                col2.metric("Symbols Covered", len(symbol_counts))
                col3.metric("Avg Relevance", f"{sum(article.get('similarity_score', 0) for article in articles) / len(articles):.2f}")
                
                # Show symbol breakdown
                if symbol_counts:
                    st.write("**Articles by Symbol:**")
                    for symbol, count in symbol_counts.items():
                        st.write(f"- {symbol}: {count} articles")
        else:
            st.info("No news data available. This is expected when using the simple news fetcher.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D;">
        <p>LLM Market Decision Agent | Built for DataTalksClub LLM Zoomcamp 2025</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

