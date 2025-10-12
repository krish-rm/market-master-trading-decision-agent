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
    """Load LLM outputs and features data."""
    try:
        # Try to use config
        outputs_file = config.LLM_OUTPUTS_FILE
        features_file = config.FEATURES_FILE
    except:
        # Fallback
        outputs_file = LLM_OUTPUTS_FILE
        features_file = FEATURES_FILE
    
    if not os.path.exists(outputs_file):
        return None, None
    
    outputs_df = pd.read_csv(outputs_file)
    outputs_df['timestamp'] = pd.to_datetime(outputs_df['timestamp'])
    
    # Load full features if available for extended history
    if os.path.exists(features_file):
        features_df = pd.read_csv(features_file)
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    else:
        features_df = None
    
    return outputs_df, features_df


def create_price_chart(df, symbol):
    """Create interactive price and indicator chart."""
    symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Trend', 'RSI & WSS', 'Volume Bias'),
        row_heights=[0.5, 0.25, 0.25]
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
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=symbol_df['timestamp'],
            y=symbol_df['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='#FF6B6B', width=2)
        ),
        row=2, col=1
    )
    
    # RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # WSS
    fig.add_trace(
        go.Scatter(
            x=symbol_df['timestamp'],
            y=symbol_df['wss'],
            mode='lines',
            name='WSS',
            line=dict(color='#4ECDC4', width=2),
            yaxis='y3'
        ),
        row=2, col=1
    )
    
    # Volume Bias
    fig.add_trace(
        go.Bar(
            x=symbol_df['timestamp'],
            y=symbol_df['volume_bias'],
            name='Volume Bias',
            marker_color='#95A5A6'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        title_text=f"{symbol} Market Analysis",
        title_font_size=20
    )
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI / WSS", row=2, col=1)
    fig.update_yaxes(title_text="Volume Bias", row=3, col=1)
    
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
    st.title("📈 LLM Market Decision Agent")
    st.markdown("""
    AI-powered market analysis using **Groq** to interpret technical indicators 
    and provide adaptive trading guidance with fast, free AI.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        outputs_df, features_df = load_data()
    
    if outputs_df is None:
        st.error("""
        ⚠️ No data found. Please run the pipeline first:
        1. `python app/fetch_data.py`
        2. `python app/compute_features.py`
        3. `python app/llm_agent.py`
        """)
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Symbol selector
    symbols = sorted(outputs_df['symbol'].unique())
    selected_symbol = st.sidebar.selectbox(
        "Select Symbol",
        symbols,
        index=0
    )
    
    # Date range
    min_date = outputs_df['timestamp'].min()
    max_date = outputs_df['timestamp'].max()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data
    filtered_df = outputs_df[
        (outputs_df['symbol'] == selected_symbol) &
        (outputs_df['timestamp'].dt.date >= date_range[0])
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
    st.sidebar.metric("Total Insights", len(filtered_df))
    
    confidence_counts = filtered_df['confidence'].value_counts()
    for conf in ['High', 'Medium', 'Low']:
        count = confidence_counts.get(conf, 0)
        st.sidebar.metric(f"{conf} Confidence", count)
    
    # Main content
    if len(filtered_df) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Charts", "🤖 Latest Insights", "📋 Data Table"])
    
    with tab1:
        st.subheader(f"{selected_symbol} Technical Analysis")
        chart = create_price_chart(filtered_df, selected_symbol)
        st.plotly_chart(chart, use_container_width=True)
        
        # Key metrics
        latest = filtered_df.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Latest Price", f"${latest['close']:.2f}")
        col2.metric("RSI", f"{latest['rsi']:.1f}")
        col3.metric("WSS", f"{latest['wss']:.2f}")
        col4.metric("Volume Bias", f"{latest['volume_bias']:.2f}x")
        col5.metric("Trend", latest['trend'].upper())
    
    with tab2:
        st.subheader("🤖 LLM-Generated Insights")
        
        # Show latest insights
        num_insights = st.slider("Number of insights to display", 1, 20, 5)
        
        for _, row in filtered_df.head(num_insights).iterrows():
            display_llm_insight(row)
    
    with tab3:
        st.subheader("📋 Complete Data Table")
        
        # Display options
        show_columns = st.multiselect(
            "Select columns to display",
            options=list(filtered_df.columns),
            default=['timestamp', 'symbol', 'close', 'rsi', 'wss', 'confidence', 'guidance']
        )
        
        if show_columns:
            st.dataframe(
                filtered_df[show_columns],
                use_container_width=True,
                height=600
            )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_symbol}_insights.csv",
            mime="text/csv"
        )
    
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

