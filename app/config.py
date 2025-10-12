"""
Configuration module for LLM Market Decision Agent.
Loads environment variables and defines project-wide constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Data Files
HOURLY_DATA_FILE = DATA_DIR / "hourly_data.csv"
FEATURES_FILE = DATA_DIR / "features.csv"
LLM_OUTPUTS_FILE = DATA_DIR / "llm_outputs.csv"

# Groq LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# Alternative models: mixtral-8x7b-32768, llama-3.1-70b-versatile, gemma2-9b-it

# Data Configuration
SYMBOLS = os.getenv("SYMBOLS", "SPY,QQQ,AAPL").split(",")
PERIOD = os.getenv("PERIOD", "7d")  # Further reduced for reliable hourly data
INTERVAL = os.getenv("INTERVAL", "1h")

# Technical Indicator Parameters
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_MA_PERIOD = 20
TREND_SMA_PERIOD = 10

# LLM Configuration
MAX_TOKENS = 500
TEMPERATURE = 0.7
MAX_RETRIES = 3

# Evaluation Configuration
REFERENCE_REASONING_SAMPLE_SIZE = 50  # For synthetic reference generation

