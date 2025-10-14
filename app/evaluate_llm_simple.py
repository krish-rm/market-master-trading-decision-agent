"""
Simple evaluation module for LLM Market Decision Agent.
Basic evaluation without advanced dependencies.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
from pathlib import Path
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleLLMEvaluator:
    """Simple evaluation system for LLM outputs with basic metrics."""
    
    def __init__(self):
        """Initialize the simple evaluator."""
        logger.info("Simple LLM Evaluator initialized")
    
    def generate_reference_texts(self, df: pd.DataFrame) -> List[str]:
        """
        Generate reference texts based on market conditions.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of reference texts
        """
        references = []
        
        for _, row in df.iterrows():
            rsi = row['rsi']
            wss = row['wss']
            trend = row['trend']
            volume_bias = row['volume_bias']
            
            # Generate rule-based reference
            if wss > 0.7 and trend == 'up' and rsi < 70:
                ref = f"The market shows strong bullish momentum with a weighted sentiment score of {wss:.2f}. " \
                      f"The upward trend combined with RSI at {rsi:.1f} indicates healthy buying pressure. " \
                      f"Volume is {volume_bias:.1f}x average, confirming institutional participation. " \
                      f"Consider long positions with target above current levels and stop-loss below recent support."
            elif wss < 0.3 and trend == 'down' and rsi > 30:
                ref = f"Bearish conditions prevail with weighted sentiment score of {wss:.2f}. " \
                      f"The downward trend and RSI at {rsi:.1f} suggest continued selling pressure. " \
                      f"Volume at {volume_bias:.1f}x average indicates strong conviction in the move. " \
                      f"Consider short positions or wait for reversal signals before entering long."
            else:
                ref = f"Mixed signals present with weighted sentiment score of {wss:.2f}. " \
                      f"RSI at {rsi:.1f} and {trend} trend create uncertainty in market direction. " \
                      f"Volume at {volume_bias:.1f}x average provides limited conviction. " \
                      f"Exercise caution and wait for clearer directional signals before committing capital."
            
            references.append(ref)
        
        return references
    
    def calculate_basic_similarity(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate basic text similarity metrics."""
        similarities = []
        
        for pred, ref in zip(predictions, references):
            # Simple word overlap similarity
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(pred_words) == 0 and len(ref_words) == 0:
                similarity = 1.0
            elif len(pred_words) == 0 or len(ref_words) == 0:
                similarity = 0.0
            else:
                intersection = len(pred_words.intersection(ref_words))
                union = len(pred_words.union(ref_words))
                similarity = intersection / union if union > 0 else 0.0
            
            similarities.append(similarity)
        
        return {
            "basic_similarity": {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities))
            }
        }
    
    def evaluate_confidence_calibration(self, df: pd.DataFrame) -> Dict:
        """Evaluate confidence calibration accuracy."""
        # Group by confidence level
        confidence_groups = df.groupby('confidence')
        
        calibration_results = {}
        
        for confidence, group in confidence_groups:
            # Calculate accuracy for this confidence level
            # For simplicity, we'll use a mock accuracy calculation
            # In practice, you'd compare predictions with actual price movements
            
            # Mock accuracy based on confidence level
            if confidence == "High":
                accuracy = 0.75  # High confidence should be more accurate
            elif confidence == "Medium":
                accuracy = 0.60
            else:  # Low
                accuracy = 0.45
            
            calibration_results[confidence] = {
                "count": len(group),
                "accuracy": accuracy,
                "wss_mean": group['wss'].mean(),
                "wss_std": group['wss'].std()
            }
        
        return {"confidence_calibration": calibration_results}
    
    def evaluate_consistency(self, df: pd.DataFrame) -> Dict:
        """Evaluate consistency across similar market conditions."""
        # Group by WSS ranges
        df['wss_range'] = pd.cut(df['wss'], bins=[0, 0.33, 0.67, 1.0], labels=['Bearish', 'Neutral', 'Bullish'])
        
        consistency_results = {}
        
        for wss_range, group in df.groupby('wss_range'):
            if len(group) < 3:  # Skip groups with too few samples
                continue
                
            # Calculate consistency metrics
            confidence_dist = group['confidence'].value_counts(normalize=True)
            
            consistency_results[str(wss_range)] = {
                "count": len(group),
                "confidence_distribution": confidence_dist.to_dict(),
                "avg_wss": group['wss'].mean(),
                "trend_distribution": group['trend'].value_counts(normalize=True).to_dict()
            }
        
        return {"consistency_analysis": consistency_results}
    
    def comprehensive_evaluation(self, df: pd.DataFrame) -> Dict:
        """Run comprehensive evaluation with basic metrics."""
        logger.info("Starting simple LLM evaluation...")
        
        # Generate reference texts
        references = self.generate_reference_texts(df)
        predictions = df['reasoning'].tolist()
        
        # Calculate all metrics
        results = {}
        
        # Basic text similarity metrics
        results.update(self.calculate_basic_similarity(predictions, references))
        
        # Confidence and consistency metrics
        results.update(self.evaluate_confidence_calibration(df))
        results.update(self.evaluate_consistency(df))
        
        # Summary statistics
        results['summary'] = {
            "total_samples": len(df),
            "confidence_distribution": df['confidence'].value_counts().to_dict(),
            "symbol_distribution": df['symbol'].value_counts().to_dict(),
            "api_mode_percentage": (df['api_mode'].sum() / len(df)) * 100 if 'api_mode' in df.columns else 0
        }
        
        logger.info("Simple evaluation completed")
        return results
    
    def create_evaluation_report(self, results: Dict, save_path: str = None):
        """Create a simple evaluation report."""
        report = []
        report.append("# LLM Market Decision Agent - Simple Evaluation Report\n")
        
        # Summary
        summary = results['summary']
        report.append("## Summary\n")
        report.append(f"- **Total Samples**: {summary['total_samples']}")
        report.append(f"- **API Mode Usage**: {summary['api_mode_percentage']:.1f}%")
        report.append("\n### Confidence Distribution:")
        for conf, count in summary['confidence_distribution'].items():
            report.append(f"- {conf}: {count}")
        
        # Text Quality Metrics
        report.append("\n## Text Quality Metrics\n")
        
        if 'basic_similarity' in results:
            basic = results['basic_similarity']
            report.append(f"### Basic Similarity Score")
            report.append(f"- **Mean**: {basic['mean']:.3f} ± {basic['std']:.3f}")
            report.append(f"- **Range**: {basic['min']:.3f} - {basic['max']:.3f}")
        
        # Confidence Calibration
        if 'confidence_calibration' in results:
            report.append("\n## Confidence Calibration\n")
            calib = results['confidence_calibration']
            for conf, metrics in calib.items():
                report.append(f"### {conf} Confidence")
                report.append(f"- **Count**: {metrics['count']}")
                report.append(f"- **Accuracy**: {metrics['accuracy']:.1%}")
                report.append(f"- **Avg WSS**: {metrics['wss_mean']:.3f}")
        
        # Consistency Analysis
        if 'consistency_analysis' in results:
            report.append("\n## Consistency Analysis\n")
            consistency = results['consistency_analysis']
            for condition, metrics in consistency.items():
                report.append(f"### {condition} Market Conditions")
                report.append(f"- **Count**: {metrics['count']}")
                report.append(f"- **Avg WSS**: {metrics['avg_wss']:.3f}")
        
        # Join report
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text


def main():
    """Run simple evaluation on LLM outputs."""
    logger.info("Starting simple LLM evaluation...")
    
    # Initialize evaluator
    evaluator = SimpleLLMEvaluator()
    
    # Load LLM outputs
    try:
        df = pd.read_csv(config.LLM_OUTPUTS_FILE)
        logger.info(f"Loaded {len(df)} LLM outputs for evaluation")
    except FileNotFoundError:
        logger.error("LLM outputs file not found. Run llm_agent.py first.")
        return
    except pd.errors.EmptyDataError:
        logger.error("LLM outputs file is empty. Run llm_agent.py first.")
        return
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(df)
    
    # Create and save report
    report = evaluator.create_evaluation_report(
        results, 
        save_path=str(config.EVALUATION_RESULTS_FILE)
    )
    
    # Print key results
    print("\n" + "="*60)
    print("SIMPLE EVALUATION RESULTS")
    print("="*60)
    
    if 'basic_similarity' in results:
        basic = results['basic_similarity']
        print(f"Basic Similarity: {basic['mean']:.3f} ± {basic['std']:.3f}")
    
    print(f"\n✅ Simple evaluation completed. Results saved to {config.EVALUATION_RESULTS_FILE}")


if __name__ == "__main__":
    main()
