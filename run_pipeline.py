"""
Complete pipeline runner for LLM Market Decision Agent.
Executes all steps in sequence: fetch → compute → analyze → evaluate
"""

import logging
import sys
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(script_path: str, description: str) -> bool:
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to Python script
        description: Human-readable description
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✅ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        logger.error(f"Error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error in {description}: {str(e)}")
        return False


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
            'required': True
        },
        {
            'script': 'app/compute_features.py',
            'description': 'Compute technical indicators and WSS',
            'required': True
        },
        {
            'script': 'app/llm_agent.py',
            'description': 'Generate LLM insights and guidance',
            'required': True
        }
    ]
    
    # Execute steps
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] Starting: {step['description']}")
        
        success = run_command(step['script'], step['description'])
        
        if not success and step['required']:
            logger.error("Pipeline stopped due to critical failure")
            logger.error("Please check logs above and fix issues before retrying")
            sys.exit(1)
    
    # Pipeline complete
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run evaluation: cd app && jupyter notebook evaluate_llm.ipynb")
    print("  2. Launch dashboard: streamlit run app/streamlit_app.py")
    print("  3. View data: Check data/ directory for CSV files")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

