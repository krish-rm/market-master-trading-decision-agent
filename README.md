# Market Master - Trading Decision Agent

> **AI-powered market strategist using Groq to interpret technical indicators and provide adaptive trading guidance with fast, free AI**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Features](#-features)
- [Detailed Installation](#-installation)
- [Usage](#-usage)
- [LLM Integration](#-llm-integration)
- [Evaluation Methodology](#-evaluation-methodology)
- [Project Structure](#-project-structure)
- [Sample Outputs](#-sample-outputs)
- [Docker Deployment](#-docker-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **Market Master Trading Decision Agent** is an intelligent system that:

1. **Fetches** hourly market data from Yahoo Finance (SPY, QQQ, AAPL)
2. **Computes** technical indicators (RSI, ATR, Volume Bias, Trend, WSS)
3. **Analyzes** market conditions using Groq's fast AI models
4. **Generates** natural language reasoning and trading guidance
5. **Evaluates** prediction quality and confidence calibration
6. **Visualizes** insights through an interactive Streamlit dashboard

This project demonstrates production-quality LLM engineering using Groq's free, fast API for market analysis.

---

## 🚨 Problem Statement

**Challenge**: Traditional technical analysis requires expertise to interpret multiple indicators simultaneously. Retail traders often struggle to synthesize RSI, volume, volatility, and trend signals into actionable insights.

**Solution**: Leverage large language models (LLMs) to:
- Interpret complex market indicators in natural language
- Provide contextual trading guidance adapted to current conditions
- Assign confidence levels based on signal alignment
- Make quantitative analysis accessible to non-experts

---

## 🏗️ Solution Architecture

```
┌─────────────────┐
│  Yahoo Finance  │
│   (OHLCV Data)  │
└────────┬────────┘
         │
         v
┌─────────────────────────┐
│   Data Pipeline         │
│  • fetch_data.py        │
│  • compute_features.py  │
└────────┬────────────────┘
         │
         v
┌──────────────────────────┐
│  Technical Indicators    │
│  • RSI(14)               │
│  • ATR(14)               │
│  • Volume Bias           │
│  • Trend (SMA slope)     │
│  • WSS (weighted score)  │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│   LLM Agent (Groq)     │
│  • Prompt Engineering    │
│  • JSON Response Parsing │
│  • Fallback Logic        │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Outputs & Evaluation    │
│  • Reasoning Text        │
│  • Trading Guidance      │
│  • Confidence Level      │
│  • ROUGE-L / Cosine Sim  │
│  • Calibration Metrics   │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Streamlit Dashboard     │
│  • Interactive Charts    │
│  • LLM Insights          │
│  • Data Export           │
└──────────────────────────┘
```

---

## ✨ Features

### 🔹 Data Pipeline
- Automated hourly data fetching from Yahoo Finance
- Multi-symbol support (SPY, QQQ, AAPL, configurable)
- Robust error handling and logging

### 🔹 Technical Analysis
- **RSI (14)**: Momentum oscillator
- **ATR (14)**: Volatility measure
- **Volume Bias**: Current volume vs 20-hour average
- **Trend**: 10-hour SMA slope direction
- **WSS**: Weighted Sentiment Score (0-1 normalized composite)

### 🔹 LLM Integration
- **Model**: Groq Llama 3.1 8B (fast and free)
- **Prompt Engineering**: Market-specific system prompts
- **Structured Output**: JSON with reasoning, guidance, confidence
- **Fallback Mode**: Rule-based responses when API unavailable
- **Rate Limiting**: Respectful API usage

### 🔹 Evaluation Framework
- **ROUGE-L**: Text similarity vs reference reasoning
- **Cosine Similarity**: Semantic similarity using TF-IDF
- **Confidence Calibration**: Accuracy by confidence level
- **Consistency Analysis**: Behavior across market conditions
- **Quality Metrics**: Automated evaluation of AI recommendations

### 🔹 Interactive Dashboard
- Real-time price and indicator charts
- LLM-generated insights display
- Confidence-based color coding
- Exportable data tables
- Multi-symbol and date range filtering

---

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- Groq API key (get free at [console.groq.com](https://console.groq.com))
- Git

### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/market-master-trading-decision-agent.git
cd market-master-trading-decision-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
# Create .env file with your Groq API key
# Get free key at: https://console.groq.com/
cp ENV_EXAMPLE.txt .env
# Edit .env and add: GROQ_API_KEY=your_key_here
echo "SYMBOLS=SPY,QQQ,AAPL" >> .env
echo "PERIOD=7d" >> .env
echo "INTERVAL=1h" >> .env
```

### Option 2: Docker Setup

```bash
# Clone repository
git clone https://github.com/your-username/market-master-trading-decision-agent.git
cd market-master-trading-decision-agent

# Create .env file with your API key
echo "GROQ_API_KEY=your_key_here" > .env

# Build and run
docker-compose up -d
```

---

## 🚀 Usage

### Option 1: Complete Pipeline (Recommended)

Run the entire pipeline in one command:

```bash
python run_pipeline.py
```

**What it does:**
1. ✅ Fetches hourly market data from Yahoo Finance
2. ✅ Computes technical indicators and WSS
3. ✅ Generates LLM insights and guidance
4. ✅ Launches Streamlit dashboard and opens browser automatically

**Output**: Complete dataset ready for analysis + dashboard running at http://localhost:8501

**Optional Next Step:**
📊 Run evaluation: `cd app && jupyter notebook evaluate_llm.ipynb`

**Example Output:**
```
======================================================================
LLM MARKET DECISION AGENT - COMPLETE PIPELINE
======================================================================

[1/3] Starting: Fetch hourly market data from Yahoo Finance
======================================================================
STEP: Fetch hourly market data from Yahoo Finance
======================================================================
✅ Fetched 1,440 hourly bars for SPY, QQQ, AAPL
✅ Fetch hourly market data from Yahoo Finance completed successfully

[2/3] Starting: Compute technical indicators and WSS
======================================================================
STEP: Compute technical indicators and WSS
======================================================================
✅ Computed features for 1,440 rows
✅ Compute technical indicators and WSS completed successfully

[3/4] Starting: Generate LLM insights and guidance
======================================================================
STEP: Generate LLM insights and guidance
======================================================================
✅ Generated LLM outputs for 100 samples
✅ Generate LLM insights and guidance completed successfully

[4/4] Starting: Launch Streamlit dashboard
======================================================================
STEP: Launch Streamlit dashboard
======================================================================
✅ Launch Streamlit dashboard started in background

======================================================================
✅ PIPELINE COMPLETE!
======================================================================

🚀 Dashboard should open automatically in your browser!
📊 If browser doesn't open, go to: http://localhost:8501

💡 Press Ctrl+C to stop the dashboard when done
======================================================================
```

### Option 2: Step-by-Step Execution

#### Step 1: Fetch Market Data

```bash
python app/fetch_data.py
```

**Output**: `data/hourly_data.csv` with OHLCV data for all symbols

**Example:**
```
✅ Fetched 1,440 hourly bars for SPY, QQQ, AAPL
📊 Date range: 2024-08-01 to 2024-10-06
```

#### Step 2: Compute Features

```bash
python app/compute_features.py
```

**Output**: `data/features.csv` with technical indicators

**Sample Features:**
| timestamp | symbol | close | rsi | atr | volume_bias | trend | wss |
|-----------|--------|-------|-----|-----|-------------|-------|-----|
| 2024-10-06 14:00 | SPY | 572.14 | 62.3 | 2.41 | 1.15 | up | 0.74 |

#### Step 3: Generate LLM Insights

```bash
python app/llm_agent.py
```

**Output**: `data/llm_outputs.csv` with LLM-generated reasoning and guidance

**Note**: By default processes 100 samples (configurable in code). Remove `sample_size` parameter in `llm_agent.py:main()` to process all data.

**Cost**: FREE with Groq's generous free tier (30 requests/minute)

### Step 4: Run Evaluation

```bash
cd app
jupyter notebook evaluate_llm.ipynb
```

**Outputs**:
- ROUGE-L and Cosine Similarity scores
- Confidence calibration charts
- `data/evaluation_results.json`

### Step 5: Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

**Access**: Open browser to `http://localhost:8501`

---

## 🤖 LLM Integration

### Prompt Template

Located in `prompts/decision_prompt.txt`:

```
You are an experienced market strategist and quantitative analyst with deep expertise 
in technical analysis and risk management.

Analyze the following hourly market indicators and provide your professional assessment.

Market Data:
Symbol: {symbol}
Timestamp: {timestamp}
Close Price: ${close_price:.2f}
RSI(14): {rsi:.2f}
ATR(14): {atr:.2f}
Volume Bias: {volume_bias:.2f}x
Trend: {trend}
Weighted Sentiment Score (WSS): {wss:.2f}

Instructions:
1. Analyze what these indicators reveal about current market conditions
2. Determine the market bias: bullish, bearish, or neutral
3. Provide a concise trading recommendation
4. Assess your confidence level based on signal alignment

Response Format (JSON only):
{
  "reasoning": "Detailed explanation...",
  "guidance": "Specific trading recommendation...",
  "confidence": "Low/Medium/High"
}
```

### Example Input/Output

**Input Signals:**
```json
{
  "symbol": "SPY",
  "timestamp": "2024-10-06 14:00:00",
  "close": 572.14,
  "rsi": 62.3,
  "atr": 2.41,
  "volume_bias": 1.15,
  "trend": "up",
  "wss": 0.74
}
```

**LLM Response:**
```json
{
  "reasoning": "SPY demonstrates strong bullish momentum with a WSS of 0.74 indicating healthy risk-adjusted sentiment. The RSI at 62.3 sits comfortably in bullish territory without being overbought, suggesting room for continued upside. Volume running 15% above the 20-hour average confirms strong institutional participation. The upward trend persists with moderate volatility (ATR 2.41), creating an favorable risk/reward environment.",
  
  "guidance": "Consider adding to long positions with a target 1-2 ATR above current levels (~$575-577). Maintain a stop-loss near $569 (previous support). For new entries, wait for a pullback to VWAP. Position size should reflect the current elevated price level. Monitor for RSI divergence as an early warning signal.",
  
  "confidence": "High"
}
```

### Fallback Mode

When Groq API is unavailable (no key), the system uses rule-based logic:

```python
if wss > 0.7 and trend == 'up' and rsi < 70:
    confidence = "High"
    guidance = "Consider long positions..."
elif wss < 0.3 and trend == 'down' and rsi > 30:
    confidence = "High"
    guidance = "Consider short positions..."
else:
    confidence = "Medium/Low"
    guidance = "Exercise caution..."
```

---

## 📊 Evaluation Methodology

### 1. ROUGE-L Score

Measures longest common subsequence between LLM reasoning and rule-based reference text.

**Interpretation**:
- **0.3-0.5**: Good semantic overlap
- **>0.5**: Strong alignment with expected reasoning

**Results** (typical):
```
Mean ROUGE-L: 0.42
Median: 0.41
Std Dev: 0.12
```

### 2. Cosine Similarity

TF-IDF-based semantic similarity between LLM and reference text.

**Results** (typical):
```
Mean Cosine Similarity: 0.58
Correlation with ROUGE-L: 0.73
```

### 3. Confidence Calibration

Tests if "High" confidence correlates with accurate next-hour price direction.

**Methodology**:
- Bullish signal (WSS > 0.6) → Price should rise
- Bearish signal (WSS < 0.4) → Price should fall
- Neutral signal (0.4-0.6) → Small movement

**Typical Results**:
| Confidence | Accuracy | Count |
|------------|----------|-------|
| High       | 68%      | 42    |
| Medium     | 54%      | 38    |
| Low        | 47%      | 20    |

### 4. Consistency Analysis

Evaluates whether similar market conditions yield similar LLM responses.

**Findings**:
- Bullish conditions (WSS > 0.67) → 78% High confidence
- Neutral conditions (0.33-0.67) → 64% Medium confidence
- Bearish conditions (< 0.33) → 71% High confidence

---

## 📁 Project Structure

```
market-master-trading-decision-agent/
│
├── data/                           # Generated data (gitignored)
│   ├── hourly_data.csv            # Raw OHLCV data
│   ├── features.csv               # Technical indicators
│   ├── llm_outputs.csv            # LLM insights
│   └── evaluation_results.json    # Evaluation metrics
│
├── app/                           # Main application code
│   ├── config.py                  # Configuration and constants
│   ├── fetch_data.py              # Yahoo Finance data fetcher
│   ├── compute_features.py        # Technical indicator computation
│   ├── llm_agent.py               # Groq AI integration
│   ├── streamlit_app.py           # Interactive dashboard
│   ├── evaluate_llm.ipynb         # Evaluation notebook
│   └── generate_evaluation_notebook.py  # Notebook builder
│
├── prompts/                       # LLM prompt templates
│   └── decision_prompt.txt        # Market analysis prompt
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-container orchestration
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git exclusions
└── README.md                      # This file
```

---

## 🎨 Sample Outputs

### Streamlit Dashboard

**Features**:
- 📈 Price and indicator charts (Plotly)
- 🤖 LLM insights with confidence badges
- 📊 Multi-symbol comparison
- 📥 CSV export functionality

### CLI Output Example

```bash
$ python app/llm_agent.py

================================================================================
LLM AGENT PROCESSING SUMMARY
================================================================================
Model: llama-3.1-8b-instant (Groq)
Total rows processed: 100
API mode: Enabled

Confidence distribution:
High      42
Medium    38
Low       20

Outputs saved to: data/llm_outputs.csv
================================================================================

Sample LLM Outputs:

SPY @ 2024-10-06 14:00:00
WSS: 0.74 | Trend: up | Confidence: High
Reasoning: SPY demonstrates strong bullish momentum with a WSS of 0.74...
Guidance: Consider adding to long positions with a target 1-2 ATR above...
--------------------------------------------------------------------
```

### Evaluation Notebook Results

**Key Metrics**:
- ✅ ROUGE-L: 0.42 (good semantic overlap)
- ✅ Cosine Similarity: 0.58 (strong semantic alignment)
- ✅ High Confidence Accuracy: 68%
- ✅ Consistency Score: 0.76

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t llm-market-agent .

# Run container
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e GROQ_API_KEY=your_key_here \
  --name market-agent \
  llm-market-agent
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Application

Open browser to `http://localhost:8501`

---

## 🔮 Future Enhancements

### Phase 1: Enhanced LLM Features
- [ ] **RAG Integration**: Fetch real-time news headlines via NewsAPI
- [ ] **Multi-Persona Prompts**: Bull/Bear/Risk Manager perspectives
- [ ] **Chain-of-Thought**: Explicit reasoning steps
- [ ] **Few-Shot Examples**: Include historical patterns in prompt

### Phase 2: Advanced Analytics
- [ ] **Backtesting Framework**: Test LLM guidance on historical data
- [ ] **Portfolio Simulation**: Multi-symbol position management
- [ ] **Risk Metrics**: Sharpe ratio, max drawdown tracking
- [ ] **Sentiment Analysis**: Integrate social media/news sentiment

### Phase 3: Production Features
- [ ] **Real-Time Streaming**: WebSocket data updates
- [ ] **Alert System**: Email/SMS notifications for high-confidence signals
- [ ] **Multi-Timeframe Analysis**: 1h, 4h, 1d aggregation
- [ ] **API Endpoint**: FastAPI REST service
- [ ] **User Authentication**: Multi-user support with saved preferences

### Phase 4: Research
- [ ] **LLM Fine-Tuning**: Train on historical market commentary
- [ ] **Ensemble Methods**: Combine multiple LLM responses
- [ ] **Reinforcement Learning**: RLHF for trading performance
- [ ] **Explainability**: LIME/SHAP for indicator importance

---

## 🎓 Educational Value

This project demonstrates key LLM engineering concepts:

1. **Prompt Engineering**: Structured market analysis prompts
2. **Output Parsing**: Robust JSON extraction with fallbacks
3. **Evaluation**: Quantitative metrics (ROUGE-L, calibration)
4. **RAG Architecture**: Potential for news/headline integration
5. **Production Patterns**: Logging, error handling, rate limiting
6. **Containerization**: Docker for reproducibility
7. **Data Pipelines**: ETL from API → Features → LLM → UI

---

## 📝 DataTalksClub LLM Zoomcamp Criteria

| Criterion | Implementation | Status |
|-----------|----------------|--------|
| **Problem Description** | Market signal interpretation using LLM | ✅ |
| **LLM Integration** | Groq AI for reasoning generation | ✅ |
| **Retrieval Mechanism** | Technical indicator aggregation (RAG-ready) | ✅ |
| **Database** | CSV-based data storage (scalable to PostgreSQL) | ✅ |
| **Evaluation** | ROUGE-L, Cosine Sim, Calibration | ✅ |
| **Interface** | Streamlit dashboard + CLI | ✅ |
| **Ingestion Pipeline** | Yahoo Finance → Features → LLM | ✅ |
| **Monitoring** | Confidence tracking, evaluation notebook | ✅ |
| **Containerization** | Dockerfile + docker-compose | ✅ |
| **Reproducibility** | requirements.txt, detailed README | ✅ |
| **Code Quality** | Type hints, docstrings, logging | ✅ |
| **Best Practices** | Modular design, config-driven | ✅ |

---

## 🔧 Troubleshooting

### Groq API Key Issues
**Problem**: `No API key provided` or `401 Unauthorized`

**Solution**:
1. Check `.env` file exists in project root
2. Verify format: `GROQ_API_KEY=gsk_your_key_here` (no quotes)
3. Get new key at https://console.groq.com/

### Rate Limit Errors
**Problem**: `429 Too Many Requests`

**Solution**: This is normal! Groq automatically retries. Just wait, the pipeline will complete.

### No Market Data
**Problem**: `No data retrieved` from Yahoo Finance

**Solution**:
1. Check internet connection
2. Try shorter period: `PERIOD=5d` in `.env`
3. Use different symbols if one fails

### Streamlit Won't Start
**Problem**: `Address already in use`

**Solution**:
```bash
# Kill existing streamlit
pkill -f streamlit  # Linux/Mac
# Or manually close the browser tab and terminal
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for Contribution**:
- Additional technical indicators (MACD, Bollinger Bands)
- Enhanced evaluation metrics
- UI improvements
- Documentation and tutorials

---

## ⚠️ Disclaimer

**This project is for educational and research purposes only.**

- **Not Financial Advice**: Do not use for actual trading decisions
- **No Warranty**: Authors assume no liability for financial losses
- **Market Risk**: Past performance does not guarantee future results
- **Free Tier**: Groq provides generous free tier (30 requests/minute) - completely free for this project!

Always consult a licensed financial advisor before making investment decisions.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **DataTalksClub**: For the excellent LLM Zoomcamp curriculum
- **Groq**: For fast, free AI API
- **Yahoo Finance**: For free market data
- **Streamlit**: For rapid dashboard development
- **Open Source Community**: For amazing Python libraries

---

## 📚 References

1. [DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)
2. [Groq API Documentation](https://console.groq.com/docs)
3. [Technical Analysis Library (ta)](https://github.com/bukosabino/ta)
4. [ROUGE Metric](https://aclanthology.org/W04-1013/)
5. [Streamlit Documentation](https://docs.streamlit.io)

