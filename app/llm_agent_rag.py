"""
Enhanced LLM Agent with RAG (Retrieval-Augmented Generation) for market analysis.
Integrates real-time news context with technical indicators for better insights.
"""

import logging
import pandas as pd
import json
import time
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from groq import Groq
import config
# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from fetch_news import NewsRetriever
    NEWS_RETRIEVER_AVAILABLE = True
except ImportError as e:
    NEWS_RETRIEVER_AVAILABLE = False
    logger.warning(f"News retriever not available: {e}. Will use mock news context.")

# Reduce noise from HTTP libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('groq._base_client').setLevel(logging.WARNING)


class RAGMarketDecisionAgent:
    """
    Enhanced market decision agent with RAG capabilities.
    Integrates real-time news context with technical analysis.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        prompt_template_path: Path = None,
        news_api_key: str = None
    ):
        """
        Initialize the RAG-enhanced Market Decision Agent.
        
        Args:
            api_key: Groq API key
            model: Groq model name
            prompt_template_path: Path to prompt template file
            news_api_key: NewsAPI key for RAG
        """
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self.provider = "groq"
        
        # Initialize Groq client
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✓ Initialized Groq with model: {self.model}")
        else:
            self.client = None
            logger.warning("No Groq API key provided. Will use fallback mode.")
        
        # Initialize news retriever for RAG
        if NEWS_RETRIEVER_AVAILABLE:
            self.news_retriever = NewsRetriever(news_api_key or config.NEWSAPI_KEY)
        else:
            self.news_retriever = None
            logger.warning("News retriever not available. Will use mock news context.")
        
        # Load prompt template
        self.prompt_template_path = (
            prompt_template_path or 
            config.PROMPTS_DIR / "decision_prompt.txt"
        )
        self.prompt_template = self._load_prompt_template()
        
        logger.info("RAG Market Decision Agent initialized")
    
    def _load_prompt_template(self) -> str:
        """Load the enhanced prompt template with RAG context."""
        try:
            with open(self.prompt_template_path, 'r') as f:
                template = f.read()
            
            # Add RAG context section to the template
            rag_section = """

Current Market News Context:
{news_context}

Instructions (Updated):
1. Analyze the technical indicators AND consider the current market news context
2. Determine how recent news might impact the technical signals
3. Provide market bias considering both technical and fundamental factors
4. Give trading recommendation that accounts for current market sentiment
5. Assess confidence based on alignment between technical signals and news sentiment

Response Format (JSON only):
{
  "reasoning": "Detailed explanation incorporating both technical indicators and news context...",
  "guidance": "Specific trading recommendation considering market sentiment...",
  "confidence": "Low/Medium/High",
  "news_influence": "Brief note on how news context influenced the decision"
}
"""
            
            # Insert RAG section before the instructions
            template = template.replace(
                "Instructions:",
                rag_section + "\nInstructions:"
            )
            
            return template
        except FileNotFoundError:
            logger.error(f"Prompt template not found: {self.prompt_template_path}")
            return self._get_fallback_prompt()
    
    def _get_fallback_prompt(self) -> str:
        """Fallback prompt if template file is not found."""
        return """You are an experienced market strategist and quantitative analyst with deep expertise in technical analysis and risk management.

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

Current Market News Context:
{news_context}

Instructions:
1. Analyze the technical indicators AND consider the current market news context
2. Determine how recent news might impact the technical signals
3. Provide market bias considering both technical and fundamental factors
4. Give trading recommendation that accounts for current market sentiment
5. Assess confidence based on alignment between technical signals and news sentiment

Response Format (JSON only):
{
  "reasoning": "Detailed explanation incorporating both technical indicators and news context...",
  "guidance": "Specific trading recommendation considering market sentiment...",
  "confidence": "Low/Medium/High",
  "news_influence": "Brief note on how news context influenced the decision"
}"""
    
    def fetch_and_index_news(self, symbols: List[str], hours_back: int = 24):
        """
        Fetch and index news for RAG retrieval.
        
        Args:
            symbols: List of stock symbols
            hours_back: Hours back to fetch news
        """
        if not self.news_retriever:
            logger.warning("News retriever not available. Skipping news indexing.")
            return
            
        logger.info(f"Fetching and indexing news for symbols: {symbols}")
        
        # Fetch news articles
        news_articles = self.news_retriever.fetch_market_news(symbols, hours_back)
        
        if not news_articles:
            logger.warning("No news articles fetched")
            return
        
        # Create embeddings and build index
        embeddings = self.news_retriever.create_embeddings(news_articles)
        self.news_retriever.build_index(news_articles, embeddings)
        
        # Save news data
        self.news_retriever.save_news_data(str(config.NEWS_DATA_FILE))
        
        logger.info(f"News indexed successfully: {len(news_articles)} articles")
    
    def retrieve_relevant_news_context(self, symbol: str, market_conditions: Dict) -> str:
        """
        Retrieve relevant news context for a specific symbol and market conditions.
        
        Args:
            symbol: Stock symbol
            market_conditions: Current market conditions dict
            
        Returns:
            Formatted news context string
        """
        # Create query based on symbol and market conditions
        query_parts = [symbol]
        
        # Add trend-based terms
        trend = market_conditions.get('trend', 'neutral')
        if trend == 'up':
            query_parts.extend(['rally', 'gains', 'bullish'])
        elif trend == 'down':
            query_parts.extend(['decline', 'losses', 'bearish'])
        
        # Add RSI-based terms
        rsi = market_conditions.get('rsi', 50)
        if rsi > 70:
            query_parts.extend(['overbought', 'resistance'])
        elif rsi < 30:
            query_parts.extend(['oversold', 'support'])
        
        query = " ".join(query_parts)
        
        # Retrieve relevant news
        if self.news_retriever:
            relevant_news = self.news_retriever.retrieve_relevant_news(query, k=3)
        else:
            # Mock news context when retriever not available
            relevant_news = [
                {
                    'title': f'Market Analysis for {symbol}',
                    'description': f'Current market conditions show {trend} trend with RSI at {rsi:.1f}.',
                    'source': 'Mock Analysis'
                }
            ]
        
        if not relevant_news:
            return "No relevant news found for current market conditions."
        
        # Format news context
        context_parts = []
        for article in relevant_news:
            similarity_score = article.get('similarity_score', 0.0)
            context_parts.append(
                f"• {article['title']} (Source: {article['source']}, "
                f"Relevance: {similarity_score:.2f})"
            )
        
        return "\n".join(context_parts)
    
    def analyze_with_rag(self, market_data: Dict) -> Dict:
        """
        Analyze market data with RAG-enhanced context.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Analysis result with RAG context
        """
        symbol = market_data['symbol']
        
        # Retrieve relevant news context
        news_context = self.retrieve_relevant_news_context(symbol, market_data)
        
        # Add news context to market data
        enhanced_data = market_data.copy()
        enhanced_data['news_context'] = news_context
        
        # Generate analysis
        if self.client:
            return self._call_groq_api(enhanced_data)
        else:
            return self._fallback_analysis(enhanced_data)
    
    def _call_groq_api(self, market_data: Dict) -> Dict:
        """Call Groq API with enhanced market data including news context."""
        try:
            # Format prompt with market data and news context
            prompt = self.prompt_template.format(**market_data)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional market analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            
            # Add metadata
            result['api_mode'] = True
            result['model'] = self.model
            result['provider'] = self.provider
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}. Using fallback.")
            return self._fallback_analysis(market_data)
        except Exception as e:
            logger.error(f"Groq API error: {e}. Using fallback.")
            return self._fallback_analysis(market_data)
    
    def _fallback_analysis(self, market_data: Dict) -> Dict:
        """Fallback analysis when API is unavailable."""
        rsi = market_data['rsi']
        wss = market_data['wss']
        trend = market_data['trend']
        news_context = market_data.get('news_context', 'No news context available')
        
        # Rule-based analysis with news consideration
        if wss > 0.7 and trend == 'up' and rsi < 70:
            confidence = "High"
            reasoning = f"Strong bullish signals with WSS {wss:.2f} and upward trend. {news_context}"
            guidance = "Consider long positions with proper risk management."
        elif wss < 0.3 and trend == 'down' and rsi > 30:
            confidence = "High"
            reasoning = f"Strong bearish signals with WSS {wss:.2f} and downward trend. {news_context}"
            guidance = "Consider short positions or wait for better entry."
        else:
            confidence = "Medium"
            reasoning = f"Mixed signals with WSS {wss:.2f} and {trend} trend. {news_context}"
            guidance = "Exercise caution and wait for clearer signals."
        
        return {
            "reasoning": reasoning,
            "guidance": guidance,
            "confidence": confidence,
            "news_influence": "Fallback analysis with news context",
            "api_mode": False,
            "model": "fallback",
            "provider": "rule_based"
        }
    
    def process_features_with_rag(self, features_df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Process features through RAG-enhanced LLM agent.
        
        Args:
            features_df: DataFrame with technical features
            sample_size: Number of samples to process (None for all)
            
        Returns:
            DataFrame with LLM outputs including RAG context
        """
        if sample_size:
            features_df = features_df.sample(n=min(sample_size, len(features_df)))
        
        logger.info(f"Processing {len(features_df)} rows through RAG-enhanced LLM agent...")
        
        results = []
        total_rows = len(features_df)
        
        for idx, row in features_df.iterrows():
            try:
                # Prepare market data
                market_data = {
                    'symbol': row['symbol'],
                    'timestamp': row['timestamp'],
                    'close_price': row['close'],
                    'rsi': row['rsi'],
                    'atr': row['atr'],
                    'volume_bias': row['volume_bias'],
                    'trend': row['trend'],
                    'wss': row['wss']
                }
                
                # Analyze with RAG
                analysis = self.analyze_with_rag(market_data)
                
                # Combine with original data
                result = row.to_dict()
                result.update({
                    'reasoning': analysis['reasoning'],
                    'guidance': analysis['guidance'],
                    'confidence': analysis['confidence'],
                    'news_influence': analysis.get('news_influence', ''),
                    'api_mode': analysis['api_mode'],
                    'model': analysis['model'],
                    'provider': analysis['provider']
                })
                
                results.append(result)
                
                # Progress logging
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{total_rows} rows...")
                
                # Rate limiting for Groq
                if analysis['api_mode']:
                    time.sleep(0.5)  # 2 requests per second max
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        logger.info(f"Processing complete: {len(results)} successful analyses")
        return results_df


def main():
    """Test the RAG-enhanced LLM agent."""
    logger.info("Testing RAG-enhanced Market Decision Agent...")
    
    # Initialize agent
    agent = RAGMarketDecisionAgent()
    
    # Load features
    try:
        features_df = pd.read_csv(config.FEATURES_FILE)
        logger.info(f"Loaded {len(features_df)} features")
    except FileNotFoundError:
        logger.error("Features file not found. Run compute_features.py first.")
        return
    
    # Fetch and index news for our symbols
    symbols = list(features_df['symbol'].unique())
    agent.fetch_and_index_news(symbols, hours_back=48)
    
    # Process a small sample
    sample_df = features_df.head(3)
    results = agent.process_features_with_rag(sample_df)
    
    # Display results
    print("\n" + "="*60)
    print("RAG-ENHANCED ANALYSIS RESULTS")
    print("="*60)
    
    for _, row in results.iterrows():
        print(f"\n{row['symbol']} @ {row['timestamp']}")
        print(f"WSS: {row['wss']:.2f} | Trend: {row['trend']} | Confidence: {row['confidence']}")
        print(f"News Influence: {row['news_influence']}")
        print(f"Reasoning: {str(row['reasoning'])[:200]}...")
        print("-" * 60)
    
    # Save results
    results.to_csv(config.LLM_OUTPUTS_FILE, index=False)
    logger.info(f"Results saved to {config.LLM_OUTPUTS_FILE}")
    
    print(f"\n✅ RAG-enhanced analysis test completed. Processed {len(results)} samples.")


if __name__ == "__main__":
    main()
