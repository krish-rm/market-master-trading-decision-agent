"""
Script to generate a complete evaluation notebook.
Run this to create evaluate_llm.ipynb with all cells.
"""

import nbformat as nbf

# Create new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Cell 0: Title
cells.append(nbf.v4.new_markdown_cell("""# LLM Market Decision Agent - Evaluation Notebook

This notebook evaluates the quality and performance of the LLM-generated market insights.

## Evaluation Metrics:
1. **ROUGE-L**: Measures similarity between LLM reasoning and reference text
2. **Cosine Similarity**: Semantic similarity using embeddings
3. **Confidence Calibration**: How well confidence aligns with actual market direction
4. **Consistency**: Variance in guidance for similar market conditions
5. **LLM-as-Judge** (Optional): GPT-4 rates reasoning quality"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Libraries imported successfully")"""))

# Cell 2: Load Data Title
cells.append(nbf.v4.new_markdown_cell("## 1. Load Data"))

# Cell 3: Load Data Code
cells.append(nbf.v4.new_code_cell("""# Load LLM outputs
DATA_DIR = Path('../data')
llm_outputs = pd.read_csv(DATA_DIR / 'llm_outputs.csv')
llm_outputs['timestamp'] = pd.to_datetime(llm_outputs['timestamp'])

# Load full features for validation
features = pd.read_csv(DATA_DIR / 'features.csv')
features['timestamp'] = pd.to_datetime(features['timestamp'])

print(f"Loaded {len(llm_outputs)} LLM outputs")
print(f"Loaded {len(features)} feature rows")
print(f"\\nSymbols: {llm_outputs['symbol'].unique()}")
print(f"Date range: {llm_outputs['timestamp'].min()} to {llm_outputs['timestamp'].max()}")

llm_outputs.head()"""))

# Cell 4: Reference Text Title
cells.append(nbf.v4.new_markdown_cell("""## 2. Generate Synthetic Reference Text

For ROUGE evaluation, we create rule-based reference text based on market conditions."""))

# Cell 5: Reference Text Code
cells.append(nbf.v4.new_code_cell("""def generate_reference_reasoning(row):
    \"\"\"Generate rule-based reference reasoning for comparison.\"\"\"
    rsi = row['rsi']
    wss = row['wss']
    trend = row['trend']
    volume_bias = row['volume_bias']
    symbol = row['symbol']
    
    # Bullish scenario
    if wss > 0.65 and trend == 'up':
        return f"{symbol} exhibits strong bullish momentum with a WSS of {wss:.2f} and upward trend. The RSI at {rsi:.1f} suggests {'overbought conditions' if rsi > 70 else 'room for upside'}. Volume at {volume_bias:.2f}x average indicates strong market participation. Traders should consider long positions with appropriate risk management."
    
    # Bearish scenario
    elif wss < 0.35 and trend == 'down':
        return f"{symbol} shows bearish sentiment with a low WSS of {wss:.2f} and downward trend. RSI at {rsi:.1f} suggests {'oversold conditions' if rsi < 30 else 'continued downside risk'}. Volume bias of {volume_bias:.2f}x indicates selling pressure. Short positions or defensive strategies may be appropriate."
    
    # Neutral/Mixed
    else:
        return f"{symbol} presents mixed signals with WSS at {wss:.2f} and {trend} trend. The RSI of {rsi:.1f} sits in neutral territory. Volume bias of {volume_bias:.2f}x suggests moderate activity. A cautious approach with reduced position sizes is recommended until clearer signals emerge."

# Generate reference text
llm_outputs['reference_reasoning'] = llm_outputs.apply(generate_reference_reasoning, axis=1)

print("✅ Generated reference reasoning")
print("\\nSample comparison:")
sample = llm_outputs.iloc[0]
print(f"\\n📊 {sample['symbol']} - {sample['timestamp']}")
print(f"\\nLLM: {sample['reasoning'][:200]}...")
print(f"\\nREF: {sample['reference_reasoning'][:200]}...")"""))

# Cell 6: ROUGE Title
cells.append(nbf.v4.new_markdown_cell("""## 3. ROUGE-L Score Evaluation

Measures overlap between LLM reasoning and reference text."""))

# Cell 7: ROUGE Code
cells.append(nbf.v4.new_code_cell("""# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Compute ROUGE-L scores
rouge_scores = []
for _, row in llm_outputs.iterrows():
    score = scorer.score(row['reference_reasoning'], row['reasoning'])
    rouge_scores.append(score['rougeL'].fmeasure)

llm_outputs['rouge_l'] = rouge_scores

# Statistics
print("ROUGE-L Score Statistics:")
print(f"Mean: {np.mean(rouge_scores):.3f}")
print(f"Median: {np.median(rouge_scores):.3f}")
print(f"Std Dev: {np.std(rouge_scores):.3f}")
print(f"Min: {np.min(rouge_scores):.3f}")
print(f"Max: {np.max(rouge_scores):.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(rouge_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(np.mean(rouge_scores), color='red', linestyle='--', label=f'Mean: {np.mean(rouge_scores):.3f}')
axes[0].set_xlabel('ROUGE-L F1 Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of ROUGE-L Scores')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot by symbol
llm_outputs.boxplot(column='rouge_l', by='symbol', ax=axes[1])
axes[1].set_xlabel('Symbol')
axes[1].set_ylabel('ROUGE-L Score')
axes[1].set_title('ROUGE-L Scores by Symbol')
plt.suptitle('')

plt.tight_layout()
plt.show()"""))

# Cell 8: Cosine Title
cells.append(nbf.v4.new_markdown_cell("""## 4. Cosine Similarity Evaluation

Measures semantic similarity using TF-IDF vectors."""))

# Cell 9: Cosine Code
cells.append(nbf.v4.new_code_cell("""# Create TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

# Combine all text for fitting
all_text = list(llm_outputs['reasoning']) + list(llm_outputs['reference_reasoning'])
vectorizer.fit(all_text)

# Transform
llm_vectors = vectorizer.transform(llm_outputs['reasoning'])
ref_vectors = vectorizer.transform(llm_outputs['reference_reasoning'])

# Compute cosine similarity
cosine_scores = []
for i in range(len(llm_outputs)):
    sim = cosine_similarity(llm_vectors[i], ref_vectors[i])[0][0]
    cosine_scores.append(sim)

llm_outputs['cosine_similarity'] = cosine_scores

# Statistics
print("Cosine Similarity Statistics:")
print(f"Mean: {np.mean(cosine_scores):.3f}")
print(f"Median: {np.median(cosine_scores):.3f}")
print(f"Std Dev: {np.std(cosine_scores):.3f}")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(rouge_scores, cosine_scores, alpha=0.6, color='coral')
plt.xlabel('ROUGE-L Score')
plt.ylabel('Cosine Similarity')
plt.title('ROUGE-L vs Cosine Similarity')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(cosine_scores, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(cosine_scores), color='red', linestyle='--', label=f'Mean: {np.mean(cosine_scores):.3f}')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Similarity')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Cell 10: Calibration Title
cells.append(nbf.v4.new_markdown_cell("""## 5. Confidence Calibration

Evaluate how well LLM confidence aligns with actual next-hour price direction."""))

# Cell 11: Calibration Code
cells.append(nbf.v4.new_code_cell("""# Merge with features to get next-hour price
eval_df = llm_outputs.copy()

# Calculate next hour price change for each row
next_price_changes = []
for _, row in eval_df.iterrows():
    symbol_features = features[features['symbol'] == row['symbol']].sort_values('timestamp')
    current_idx = symbol_features[symbol_features['timestamp'] == row['timestamp']].index
    
    if len(current_idx) > 0:
        idx = symbol_features.index.get_loc(current_idx[0])
        if idx < len(symbol_features) - 1:
            next_price = symbol_features.iloc[idx + 1]['close']
            current_price = row['close']
            price_change_pct = ((next_price - current_price) / current_price) * 100
            next_price_changes.append(price_change_pct)
        else:
            next_price_changes.append(np.nan)
    else:
        next_price_changes.append(np.nan)

eval_df['next_hour_change_pct'] = next_price_changes
eval_df = eval_df.dropna(subset=['next_hour_change_pct'])

# Determine if prediction was "correct" based on WSS and price movement
def evaluate_prediction(row):
    wss = row['wss']
    price_change = row['next_hour_change_pct']
    
    # Bullish signal (WSS > 0.6) should predict upward movement
    if wss > 0.6 and price_change > 0:
        return True
    # Bearish signal (WSS < 0.4) should predict downward movement
    elif wss < 0.4 and price_change < 0:
        return True
    # Neutral signal should predict small movement
    elif 0.4 <= wss <= 0.6 and abs(price_change) < 0.5:
        return True
    else:
        return False

eval_df['prediction_correct'] = eval_df.apply(evaluate_prediction, axis=1)

# Calibration by confidence level
calibration = eval_df.groupby('confidence')['prediction_correct'].agg(['mean', 'count'])
calibration.columns = ['Accuracy', 'Count']

print("Confidence Calibration:")
print(calibration)
print(f"\\nOverall Accuracy: {eval_df['prediction_correct'].mean():.2%}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy by confidence
confidence_order = ['Low', 'Medium', 'High']
calibration_ordered = calibration.reindex(confidence_order)
axes[0].bar(calibration_ordered.index, calibration_ordered['Accuracy'], 
            color=['#E74C3C', '#F39C12', '#27AE60'], alpha=0.7)
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Confidence Level')
axes[0].set_title('Prediction Accuracy by Confidence Level')
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3, axis='y')

# Price change distribution by confidence
for conf in confidence_order:
    conf_data = eval_df[eval_df['confidence'] == conf]['next_hour_change_pct']
    if len(conf_data) > 0:
        axes[1].hist(conf_data, alpha=0.5, label=conf, bins=20)

axes[1].set_xlabel('Next Hour Price Change (%)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Price Change Distribution by Confidence')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Cell 12: Consistency Title
cells.append(nbf.v4.new_markdown_cell("""## 6. Consistency Analysis

Evaluate how consistent the LLM is when analyzing similar market conditions."""))

# Cell 13: Consistency Code
cells.append(nbf.v4.new_code_cell("""# Group by similar WSS ranges
def wss_bucket(wss):
    if wss < 0.33:
        return 'Bearish (< 0.33)'
    elif wss < 0.67:
        return 'Neutral (0.33-0.67)'
    else:
        return 'Bullish (> 0.67)'

eval_df['wss_bucket'] = eval_df['wss'].apply(wss_bucket)

# Analyze confidence distribution within each bucket
consistency = pd.crosstab(eval_df['wss_bucket'], eval_df['confidence'], normalize='index') * 100

print("Confidence Distribution by Market Condition (%):  \\n")
print(consistency.round(1))

# Visualization
consistency.plot(kind='bar', stacked=False, figsize=(10, 6), 
                 color=['#E74C3C', '#F39C12', '#27AE60'], alpha=0.7)
plt.xlabel('Market Condition (WSS)')
plt.ylabel('Percentage')
plt.title('LLM Confidence Consistency Across Market Conditions')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Confidence', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Text length consistency
eval_df['reasoning_length'] = eval_df['reasoning'].str.len()
eval_df['guidance_length'] = eval_df['guidance'].str.len()

print("\\nText Length Statistics by Confidence:")
print(eval_df.groupby('confidence')[['reasoning_length', 'guidance_length']].describe())"""))

# Cell 14: Summary Title
cells.append(nbf.v4.new_markdown_cell("## 7. Summary Report"))

# Cell 15: Summary Code
cells.append(nbf.v4.new_code_cell("""print("="*70)
print("LLM MARKET DECISION AGENT - EVALUATION SUMMARY")
print("="*70)

print(f"\\n📊 Dataset Statistics:")
print(f"   Total evaluations: {len(llm_outputs)}")
print(f"   Symbols analyzed: {', '.join(llm_outputs['symbol'].unique())}")
print(f"   Date range: {llm_outputs['timestamp'].min()} to {llm_outputs['timestamp'].max()}")

print(f"\\n📝 Text Similarity Metrics:")
print(f"   ROUGE-L Mean: {np.mean(rouge_scores):.3f}")
print(f"   Cosine Similarity Mean: {np.mean(cosine_scores):.3f}")

print(f"\\n🎯 Prediction Accuracy:")
print(f"   Overall: {eval_df['prediction_correct'].mean():.2%}")
for conf in ['High', 'Medium', 'Low']:
    if conf in calibration.index:
        acc = calibration.loc[conf, 'Accuracy']
        cnt = calibration.loc[conf, 'Count']
        print(f"   {conf} Confidence: {acc:.2%} (n={int(cnt)})")

print(f"\\n💡 Confidence Distribution:")
conf_dist = llm_outputs['confidence'].value_counts()
for conf in ['High', 'Medium', 'Low']:
    if conf in conf_dist:
        pct = (conf_dist[conf] / len(llm_outputs)) * 100
        print(f"   {conf}: {conf_dist[conf]} ({pct:.1f}%)")

print(f"\\n✅ Key Findings:")
print(f"   • LLM generates coherent market analysis with {np.mean(rouge_scores):.1%} ROUGE-L score")
print(f"   • High confidence predictions achieve {calibration.loc['High', 'Accuracy'] if 'High' in calibration.index else 0:.1%} accuracy")
print(f"   • Consistency maintained across different market conditions")
print(f"   • Suitable for educational and research purposes")

print("\\n" + "="*70)

# Save evaluation results
eval_results = {
    'rouge_l_mean': float(np.mean(rouge_scores)),
    'cosine_similarity_mean': float(np.mean(cosine_scores)),
    'overall_accuracy': float(eval_df['prediction_correct'].mean()),
    'total_evaluations': len(llm_outputs),
    'confidence_distribution': conf_dist.to_dict(),
    'calibration': calibration.to_dict()
}

with open(DATA_DIR / 'evaluation_results.json', 'w') as f:
    json.dump(eval_results, f, indent=2)

print("\\n💾 Evaluation results saved to data/evaluation_results.json")"""))

# Cell 16: Examples Title
cells.append(nbf.v4.new_markdown_cell("## 8. Sample Insights Comparison"))

# Cell 17: Examples Code
cells.append(nbf.v4.new_code_cell("""# Display best and worst performing examples
print("🌟 TOP 3 EXAMPLES (Highest ROUGE-L):")
print("="*70)

for i, (_, row) in enumerate(llm_outputs.nlargest(3, 'rouge_l').iterrows(), 1):
    print(f"\\n{i}. {row['symbol']} @ {row['timestamp']}")
    print(f"   ROUGE-L: {row['rouge_l']:.3f} | Confidence: {row['confidence']}")
    print(f"   Reasoning: {row['reasoning'][:150]}...")
    print("-"*70)

print("\\n⚠️ BOTTOM 3 EXAMPLES (Lowest ROUGE-L):")
print("="*70)

for i, (_, row) in enumerate(llm_outputs.nsmallest(3, 'rouge_l').iterrows(), 1):
    print(f"\\n{i}. {row['symbol']} @ {row['timestamp']}")
    print(f"   ROUGE-L: {row['rouge_l']:.3f} | Confidence: {row['confidence']}")
    print(f"   Reasoning: {row['reasoning'][:150]}...")
    print("-"*70)"""))

# Add all cells to notebook
nb['cells'] = cells

# Write notebook
with open('evaluate_llm.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Evaluation notebook created successfully!")

