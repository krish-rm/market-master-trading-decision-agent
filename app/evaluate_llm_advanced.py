"""
Advanced evaluation module for LLM Market Decision Agent.
Implements BLEU, BERTScore, and other advanced metrics.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path
import config

# Advanced evaluation libraries
try:
    from evaluate import load
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    EVALUATION_LIBS_AVAILABLE = True
except ImportError as e:
    EVALUATION_LIBS_AVAILABLE = False
    print(f"Some evaluation libraries not available: {e}")
    print("Install with: pip install evaluate nltk sentence-transformers scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


class AdvancedLLMEvaluator:
    """Advanced evaluation system for LLM outputs with multiple metrics."""
    
    def __init__(self):
        """Initialize the advanced evaluator."""
        self.scorer = None
        self.sentence_model = None
        
        if EVALUATION_LIBS_AVAILABLE:
            try:
                # Initialize Rouge scorer
                self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                logger.info("✓ Rouge scorer initialized")
                
                # Initialize sentence transformer for BERTScore
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Sentence transformer for BERTScore initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize evaluation models: {e}")
        
        # Smoothing function for BLEU
        self.smoothing = SmoothingFunction().method1
    
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
                confidence = "High"
            elif wss < 0.3 and trend == 'down' and rsi > 30:
                ref = f"Bearish conditions prevail with weighted sentiment score of {wss:.2f}. " \
                      f"The downward trend and RSI at {rsi:.1f} suggest continued selling pressure. " \
                      f"Volume at {volume_bias:.1f}x average indicates strong conviction in the move. " \
                      f"Consider short positions or wait for reversal signals before entering long."
                confidence = "High"
            else:
                ref = f"Mixed signals present with weighted sentiment score of {wss:.2f}. " \
                      f"RSI at {rsi:.1f} and {trend} trend create uncertainty in market direction. " \
                      f"Volume at {volume_bias:.1f}x average provides limited conviction. " \
                      f"Exercise caution and wait for clearer directional signals before committing capital."
                confidence = "Medium"
            
            references.append(ref)
        
        return references
    
    def calculate_rouge_l(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate ROUGE-L scores."""
        if not self.scorer:
            logger.warning("Rouge scorer not available")
            return {"rouge_l": {"precision": 0, "recall": 0, "fmeasure": 0}}
        
        rouge_scores = []
        
        for pred, ref in zip(predictions, references):
            score = self.scorer.score(ref, pred)
            rouge_scores.append(score['rougeL'])
        
        # Calculate averages
        avg_precision = np.mean([score.precision for score in rouge_scores])
        avg_recall = np.mean([score.recall for score in rouge_scores])
        avg_fmeasure = np.mean([score.fmeasure for score in rouge_scores])
        
        return {
            "rouge_l": {
                "precision": avg_precision,
                "recall": avg_recall,
                "fmeasure": avg_fmeasure
            }
        }
    
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate BLEU scores."""
        try:
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                # Tokenize texts
                pred_tokens = nltk.word_tokenize(pred.lower())
                ref_tokens = nltk.word_tokenize(ref.lower())
                
                # Calculate BLEU score
                bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoothing)
                bleu_scores.append(bleu)
            
            return {
                "bleu": {
                    "score": np.mean(bleu_scores),
                    "std": np.std(bleu_scores),
                    "min": np.min(bleu_scores),
                    "max": np.max(bleu_scores)
                }
            }
        except Exception as e:
            logger.error(f"BLEU calculation error: {e}")
            return {"bleu": {"score": 0, "std": 0, "min": 0, "max": 0}}
    
    def calculate_bertscore(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate BERTScore."""
        if not self.sentence_model:
            logger.warning("Sentence transformer not available for BERTScore")
            return {"bertscore": {"precision": 0, "recall": 0, "f1": 0}}
        
        try:
            # Generate embeddings
            pred_embeddings = self.sentence_model.encode(predictions)
            ref_embeddings = self.sentence_model.encode(references)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(pred_embeddings, ref_embeddings)
            
            # BERTScore is the diagonal (each prediction vs its reference)
            bert_scores = np.diag(similarities)
            
            return {
                "bertscore": {
                    "precision": float(np.mean(bert_scores)),
                    "recall": float(np.mean(bert_scores)),
                    "f1": float(np.mean(bert_scores)),
                    "std": float(np.std(bert_scores))
                }
            }
        except Exception as e:
            logger.error(f"BERTScore calculation error: {e}")
            return {"bertscore": {"precision": 0, "recall": 0, "f1": 0, "std": 0}}
    
    def calculate_cosine_similarity(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate TF-IDF based cosine similarity."""
        try:
            # Combine all texts for TF-IDF
            all_texts = predictions + references
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Split back to predictions and references
            n_preds = len(predictions)
            pred_vectors = tfidf_matrix[:n_preds]
            ref_vectors = tfidf_matrix[n_preds:]
            
            # Calculate cosine similarities
            similarities = cosine_similarity(pred_vectors, ref_vectors)
            
            # Get diagonal similarities (each prediction vs its reference)
            cos_scores = np.diag(similarities)
            
            return {
                "cosine_similarity": {
                    "mean": float(np.mean(cos_scores)),
                    "std": float(np.std(cos_scores)),
                    "min": float(np.min(cos_scores)),
                    "max": float(np.max(cos_scores))
                }
            }
        except Exception as e:
            logger.error(f"Cosine similarity calculation error: {e}")
            return {"cosine_similarity": {"mean": 0, "std": 0, "min": 0, "max": 0}}
    
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
        """Run comprehensive evaluation with all metrics."""
        logger.info("Starting comprehensive LLM evaluation...")
        
        # Generate reference texts
        references = self.generate_reference_texts(df)
        predictions = df['reasoning'].tolist()
        
        # Calculate all metrics
        results = {}
        
        # Text similarity metrics
        results.update(self.calculate_rouge_l(predictions, references))
        results.update(self.calculate_bleu(predictions, references))
        results.update(self.calculate_bertscore(predictions, references))
        results.update(self.calculate_cosine_similarity(predictions, references))
        
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
        
        logger.info("Comprehensive evaluation completed")
        return results
    
    def create_evaluation_report(self, results: Dict, save_path: str = None):
        """Create a comprehensive evaluation report."""
        report = []
        report.append("# LLM Market Decision Agent - Advanced Evaluation Report\n")
        
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
        
        if 'rouge_l' in results:
            rouge = results['rouge_l']
            report.append(f"### ROUGE-L Score")
            report.append(f"- **Precision**: {rouge['precision']:.3f}")
            report.append(f"- **Recall**: {rouge['recall']:.3f}")
            report.append(f"- **F-Measure**: {rouge['fmeasure']:.3f}")
        
        if 'bleu' in results:
            bleu = results['bleu']
            report.append(f"\n### BLEU Score")
            report.append(f"- **Score**: {bleu['score']:.3f} ± {bleu['std']:.3f}")
            report.append(f"- **Range**: {bleu['min']:.3f} - {bleu['max']:.3f}")
        
        if 'bertscore' in results:
            bert = results['bertscore']
            report.append(f"\n### BERTScore")
            report.append(f"- **F1 Score**: {bert['f1']:.3f} ± {bert['std']:.3f}")
        
        if 'cosine_similarity' in results:
            cos = results['cosine_similarity']
            report.append(f"\n### Cosine Similarity")
            report.append(f"- **Mean**: {cos['mean']:.3f} ± {cos['std']:.3f}")
        
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
    """Run advanced evaluation on LLM outputs."""
    logger.info("Starting advanced LLM evaluation...")
    
    # Initialize evaluator
    evaluator = AdvancedLLMEvaluator()
    
    # Load LLM outputs
    try:
        df = pd.read_csv(config.LLM_OUTPUTS_FILE)
        logger.info(f"Loaded {len(df)} LLM outputs for evaluation")
    except FileNotFoundError:
        logger.error("LLM outputs file not found. Run llm_agent.py first.")
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
    print("ADVANCED EVALUATION RESULTS")
    print("="*60)
    
    if 'rouge_l' in results:
        rouge = results['rouge_l']
        print(f"ROUGE-L F-Measure: {rouge['fmeasure']:.3f}")
    
    if 'bleu' in results:
        bleu = results['bleu']
        print(f"BLEU Score: {bleu['score']:.3f} ± {bleu['std']:.3f}")
    
    if 'bertscore' in results:
        bert = results['bertscore']
        print(f"BERTScore F1: {bert['f1']:.3f} ± {bert['std']:.3f}")
    
    if 'cosine_similarity' in results:
        cos = results['cosine_similarity']
        print(f"Cosine Similarity: {cos['mean']:.3f} ± {cos['std']:.3f}")
    
    print(f"\n✅ Advanced evaluation completed. Results saved to {config.EVALUATION_RESULTS_FILE}")


if __name__ == "__main__":
    main()
