"""
Advanced pipeline runner for LLM Market Decision Agent with RAG and Multi-timeframe support.
Executes enhanced pipeline: fetch → compute → fetch_news → analyze_with_rag → evaluate_advanced
"""

import logging
import sys
from pathlib import Path
import subprocess
import webbrowser
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(script_path: str, description: str, background: bool = False) -> bool:
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to Python script
        description: Human-readable description
        background: If True, run in background (for Streamlit)
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*70}")
    
    try:
        if background:
            # For Streamlit, run in background and don't wait
            if script_path.startswith('streamlit'):
                # Split streamlit command into parts
                cmd_parts = script_path.split()
                process = subprocess.Popen(
                    cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            logger.info(f"✅ {description} started in background\n")
            return True, process
        else:
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=False,
                text=True
            )
            logger.info(f"✅ {description} completed successfully\n")
            return True, None
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        logger.error(f"Error: {e}")
        return False, None
    except Exception as e:
        logger.error(f"❌ Unexpected error in {description}: {str(e)}")
        return False, None


def open_browser_delayed(url: str, delay: int = 5):
    """Open browser after a delay to allow Streamlit to start."""
    time.sleep(delay)
    try:
        # Only open if not already open (basic check)
        import requests
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ Dashboard already running at {url}")
                return
        except:
            pass  # If we can't check, proceed with opening
        
        webbrowser.open(url)
        print(f"🚀 Opening dashboard: {url}")
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")
        print(f"📊 Please manually open: {url}")


def main():
    """Execute complete LLM Market Decision Agent pipeline with advanced features."""
    print("\n" + "="*70)
    print("LLM MARKET DECISION AGENT - COMPLETE PIPELINE")
    print("Features: Real LLM Analysis + Multi-timeframe + RAG + Advanced Evaluation")
    print("="*70 + "\n")
    
    # Define advanced pipeline steps
    steps = [
        {
            'script': 'app/fetch_data.py',
            'description': 'Fetch hourly market data from Yahoo Finance',
            'required': True,
            'background': False
        },
        {
            'script': 'app/fetch_data_multi_timeframe.py',
            'description': 'Fetch multi-timeframe data (4h, 1d, 1w) for comparative analysis',
            'required': False,  # Optional step
            'background': False
        },
        {
            'script': 'app/compute_features.py',
            'description': 'Compute technical indicators and WSS',
            'required': True,
            'background': False
        },
        {
            'script': 'app/fetch_news_simple.py',
            'description': 'Fetch and index market news for RAG (simple version)',
            'required': False,  # Optional if no NewsAPI key
            'background': False
        },
        {
            'script': 'app/llm_agent.py',
            'description': 'Generate comprehensive LLM insights for 1-hour data (real API calls)',
            'required': True,
            'background': False
        },
        {
            'script': 'app/evaluate_llm_simple.py',
            'description': 'Run simple evaluation with basic metrics',
            'required': False,  # Optional evaluation step
            'background': False
        },
        {
            'script': 'streamlit run app/streamlit_app.py',
            'description': 'Launch Streamlit dashboard',
            'required': False,
            'background': True
        }
    ]
    
    # Execute steps
    streamlit_process = None
    completed_steps = 0
    total_steps = len(steps)
    
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{total_steps}] Starting: {step['description']}")
        
        if step['script'].startswith('streamlit'):
            # Handle Streamlit launch specially
            success, streamlit_process = run_command(step['script'], step['description'], background=True)
            if success:
                # Start browser opening in a separate thread
                browser_thread = threading.Thread(target=open_browser_delayed, args=('http://localhost:8501',))
                browser_thread.daemon = True
                browser_thread.start()
                completed_steps += 1
        else:
            success, _ = run_command(step['script'], step['description'], background=False)
            if success:
                completed_steps += 1
            
        if not success and step['required']:
            logger.error("Pipeline stopped due to critical failure")
            logger.error("Please check logs above and fix issues before retrying")
            if streamlit_process:
                streamlit_process.terminate()
            sys.exit(1)
        elif not success and not step['required']:
            logger.warning(f"Optional step failed: {step['description']}")
    
    # Pipeline complete
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("="*70)
    print(f"\n📊 Completed {completed_steps}/{total_steps} steps successfully")
    
    if streamlit_process:
        print("\n🚀 Dashboard should open automatically in your browser!")
        print("📊 If browser doesn't open, go to: http://localhost:8501")
    
    print("\n📈 Enhanced Features Available:")
    print("  • Comprehensive LLM analysis with real API calls (30 samples)")
    print("  • Multi-timeframe data analysis (1h, 4h, 1d, 1w)")
    print("  • RAG-enhanced market analysis with news context")
    print("  • Advanced evaluation metrics (BLEU, BERTScore)")
    print("  • Authentic AI insights for 1-hour data")
    
    print("\n📁 Generated Files:")
    print("  • data/hourly_data.csv - 1-hour market data")
    print("  • data/4hourly_data.csv - 4-hour market data")
    print("  • data/daily_data.csv - Daily market data")
    print("  • data/weekly_data.csv - Weekly market data")
    print("  • data/features.csv - Technical indicators")
    print("  • data/news_data.json - Market news for RAG")
    print("  • data/llm_outputs.csv - RAG-enhanced LLM analysis")
    print("  • data/evaluation_results.json - Advanced evaluation metrics")
    print("  • data/timeframe_comparison.csv - Multi-timeframe comparison")
    
    print("\n💡 Press Ctrl+C to stop the dashboard when done")
    print("="*70 + "\n")
    
    # Keep the script running to maintain the Streamlit process
    try:
        if streamlit_process:
            streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping dashboard...")
        if streamlit_process:
            streamlit_process.terminate()
        print("✅ Dashboard stopped. Goodbye!")


if __name__ == "__main__":
    main()
