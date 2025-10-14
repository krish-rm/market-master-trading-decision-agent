"""
Complete pipeline runner for LLM Market Decision Agent.
Executes all steps in sequence: fetch → compute → analyze → evaluate
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
    webbrowser.open(url)


def main():
    """Execute complete pipeline."""
    print("\n" + "="*70)
    print("LLM MARKET DECISION AGENT - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # Define pipeline steps
    steps = [
        {
            'script': 'app/fetch_data.py',
            'description': 'Fetch hourly market data from Yahoo Finance',
            'required': True,
            'background': False
        },
        {
            'script': 'app/compute_features.py',
            'description': 'Compute technical indicators and WSS',
            'required': True,
            'background': False
        },
        {
            'script': 'app/llm_agent.py',
            'description': 'Generate LLM insights and guidance',
            'required': True,
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
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] Starting: {step['description']}")
        
        if step['script'].startswith('streamlit'):
            # Handle Streamlit launch specially
            success, streamlit_process = run_command(step['script'], step['description'], background=True)
            if success:
                # Start browser opening in a separate thread
                browser_thread = threading.Thread(target=open_browser_delayed, args=('http://localhost:8501',))
                browser_thread.daemon = True
                browser_thread.start()
        else:
            success, _ = run_command(step['script'], step['description'], background=False)
            
        if not success and step['required']:
            logger.error("Pipeline stopped due to critical failure")
            logger.error("Please check logs above and fix issues before retrying")
            if streamlit_process:
                streamlit_process.terminate()
            sys.exit(1)
    
    # Pipeline complete
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\n🚀 Dashboard should open automatically in your browser!")
    print("📊 If browser doesn't open, go to: http://localhost:8501")
    print("\nOther options:")
    print("  • Run evaluation: cd app && jupyter notebook evaluate_llm.ipynb")
    print("  • View data: Check data/ directory for CSV files")
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

