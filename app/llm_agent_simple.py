"""
Simple LLM Agent that works without RAG dependencies.
Fallback version for when advanced dependencies are not available.
"""

import logging
import pandas as pd
import json
import time
from typing import Dict, Tuple, Optional
from pathlib import Path
from groq import Groq
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from HTTP libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('groq._base_client').setLevel(logging.WARNING)


class SimpleMarketDecisionAgent:
    """
    Simple market decision agent without RAG dependencies.
    Provides basic LLM analysis with mock news context.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        prompt_template_path: Path = None
    ):
        """
        Initialize the Simple Market Decision Agent.
        
        Args:
            api_key: Groq API key
            model: Groq model name
            prompt_template_path: Path to prompt template file
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
        
        # Load prompt template
        self.prompt_template_path = (
            prompt_template_path or 
            config.PROMPTS_DIR / "decision_prompt.txt"
        )
        self.prompt_template = self._load_prompt_template()
        
        logger.info("Simple Market Decision Agent initialized")
    
    def _load_prompt_template(self) -> str:
        """Load the enhanced prompt template with mock news context."""
        try:
            with open(self.prompt_template_path, 'r') as f:
                template = f.read()
            
            # Add mock news context section to the template
            mock_news_section = """

Current Market News Context:
{mock_news_context}

Instructions (Updated):
1. Analyze the technical indicators AND consider the current market context
2. Determine how current market sentiment might impact the technical signals
3. Provide market bias considering both technical and contextual factors
4. Give trading recommendation that accounts for current market sentiment
5. Assess confidence based on alignment between technical signals and market context

Response Format (JSON only):
{
  "reasoning": "Detailed explanation incorporating both technical indicators and market context...",
  "guidance": "Specific trading recommendation considering market sentiment...",
  "confidence": "Low/Medium/High",
  "news_influence": "Brief note on how market context influenced the decision"
}
"""
            
            # Insert mock news section before the instructions
            template = template.replace(
                "Instructions:",
                mock_news_section + "\nInstructions:"
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
{mock_news_context}

Instructions:
1. Analyze the technical indicators AND consider the current market context
2. Determine how current market sentiment might impact the technical signals
3. Provide market bias considering both technical and contextual factors
4. Give trading recommendation that accounts for current market sentiment
5. Assess confidence based on alignment between technical signals and market context

Response Format (JSON only):
{
  "reasoning": "Detailed explanation incorporating both technical indicators and market context...",
  "guidance": "Specific trading recommendation considering market sentiment...",
  "confidence": "Low/Medium/High",
  "news_influence": "Brief note on how market context influenced the decision"
}"""
    
    def generate_mock_news_context(self, symbol: str, market_conditions: Dict) -> str:
        """
        Generate mock news context based on market conditions.
        
        Args:
            symbol: Stock symbol
            market_conditions: Current market conditions dict
            
        Returns:
            Mock news context string
        """
        trend = market_conditions.get('trend', 'neutral')
        rsi = market_conditions.get('rsi', 50)
        wss = market_conditions.get('wss', 0.5)
        
        # Generate contextual news based on market conditions
        if trend == 'up' and rsi > 50:
            context = f"• {symbol} shows strong upward momentum with institutional buying pressure"
            context += f"• Market analysts expect continued bullish sentiment for {symbol}"
            context += f"• Recent earnings reports and sector performance support positive outlook"
        elif trend == 'down' and rsi < 50:
            context = f"• {symbol} faces selling pressure with bearish sentiment prevailing"
            context += f"• Market concerns about sector headwinds affecting {symbol} performance"
            context += f"• Technical indicators suggest caution for {symbol} positions"
        else:
            context = f"• {symbol} trading in consolidation phase with mixed signals"
            context += f"• Market waiting for clearer directional signals for {symbol}"
            context += f"• Balanced risk-reward profile for {symbol} at current levels"
        
        return context
    
    def analyze_with_mock_context(self, market_data: Dict) -> Dict:
        """
        Analyze market data with mock news context.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Analysis result with mock context
        """
        symbol = market_data['symbol']
        
        # Generate mock news context
        mock_news_context = self.generate_mock_news_context(symbol, market_data)
        
        # Add mock news context to market data
        enhanced_data = market_data.copy()
        enhanced_data['mock_news_context'] = mock_news_context
        
        # Generate analysis
        if self.client:
            return self._call_groq_api(enhanced_data)
        else:
            return self._fallback_analysis(enhanced_data)
    
    def _call_groq_api(self, market_data: Dict) -> Dict:
        """Call Groq API with enhanced market data including mock news context."""
        try:
            # Format prompt with market data and mock news context
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
        symbol = market_data['symbol']
        close_price = market_data['close_price']
        volume_bias = market_data['volume_bias']
        
        # Symbol-specific analysis templates
        symbol_contexts = {
            'AAPL': {
                'company': 'Apple Inc.',
                'sector': 'technology',
                'characteristics': 'strong institutional holding, high volatility, earnings sensitivity'
            },
            'QQQ': {
                'company': 'NASDAQ-100 ETF',
                'sector': 'technology-heavy',
                'characteristics': 'tech sector proxy, high beta, growth-oriented'
            },
            'SPY': {
                'company': 'S&P 500 ETF',
                'sector': 'broad market',
                'characteristics': 'market benchmark, lower volatility, diversified exposure'
            }
        }
        
        symbol_info = symbol_contexts.get(symbol, {
            'company': symbol,
            'sector': 'financial',
            'characteristics': 'market instrument'
        })
        
        # More sophisticated rule-based analysis
        if wss > 0.6 and trend == 'up' and rsi < 70:
            confidence = "High"
            if symbol == 'AAPL':
                reasoning = f"Strong bullish momentum for {symbol_info['company']} with WSS {wss:.2f}. Tech sector showing institutional buying pressure. Volume bias {volume_bias:.2f}x indicates strong participation."
                guidance = "Consider long positions with tight stop-loss below recent lows."
            elif symbol == 'QQQ':
                reasoning = f"QQQ showing robust tech sector leadership with WSS {wss:.2f}. NASDAQ momentum building with {symbol_info['characteristics']}. Volume confirmation at {volume_bias:.2f}x."
                guidance = "QQQ long positions favored, monitor tech sector rotation."
            else:  # SPY
                reasoning = f"SPY displaying broad market strength with WSS {wss:.2f}. S&P 500 showing institutional accumulation. Volume at {volume_bias:.2f}x supports the move."
                guidance = "SPY long positions suitable for market exposure."
                
        elif wss < 0.4 and trend == 'down' and rsi > 30:
            confidence = "High"
            if symbol == 'AAPL':
                reasoning = f"Apple facing selling pressure with WSS {wss:.2f}. Tech sector headwinds affecting {symbol_info['characteristics']}. Volume {volume_bias:.2f}x shows distribution."
                guidance = "Avoid long positions, consider protective puts."
            elif symbol == 'QQQ':
                reasoning = f"QQQ showing tech sector weakness with WSS {wss:.2f}. NASDAQ underperforming due to {symbol_info['characteristics']}. Volume {volume_bias:.2f}x indicates selling."
                guidance = "QQQ short positions or wait for better entry."
            else:  # SPY
                reasoning = f"SPY showing broad market weakness with WSS {wss:.2f}. S&P 500 experiencing selling pressure. Volume {volume_bias:.2f}x confirms distribution."
                guidance = "SPY short positions or defensive positioning."
                
        else:
            confidence = "Medium"
            if symbol == 'AAPL':
                reasoning = f"Apple in consolidation with mixed signals (WSS {wss:.2f}, {trend} trend). {symbol_info['characteristics']} creating uncertainty. Volume {volume_bias:.2f}x shows indecision."
                guidance = "Wait for clearer directional signals before positioning."
            elif symbol == 'QQQ':
                reasoning = f"QQQ showing mixed signals with WSS {wss:.2f} and {trend} trend. Tech sector {symbol_info['characteristics']} creating volatility. Volume {volume_bias:.2f}x indicates caution."
                guidance = "QQQ range-bound, wait for breakout confirmation."
            else:  # SPY
                reasoning = f"SPY in neutral territory with WSS {wss:.2f} and {trend} trend. Market {symbol_info['characteristics']} showing mixed signals. Volume {volume_bias:.2f}x suggests consolidation."
                guidance = "SPY range trading, wait for directional confirmation."
        
        return {
            "reasoning": reasoning,
            "guidance": guidance,
            "confidence": confidence,
            "news_influence": "Mock context analysis with market sentiment",
            "api_mode": False,
            "model": "fallback",
            "provider": "rule_based"
        }
    
    def process_features(self, features_df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Process features through simple LLM agent.
        
        Args:
            features_df: DataFrame with technical features
            sample_size: Number of samples to process (None for all)
            
        Returns:
            DataFrame with LLM outputs including mock context
        """
        if sample_size:
            features_df = features_df.sample(n=min(sample_size, len(features_df)))
        
        logger.info(f"Processing {len(features_df)} rows through simple LLM agent...")
        
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
                
                # Analyze with mock context
                analysis = self.analyze_with_mock_context(market_data)
                
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
    """Test the simple LLM agent."""
    logger.info("Testing Simple Market Decision Agent...")
    
    # Initialize agent
    agent = SimpleMarketDecisionAgent()
    
    # Load features
    try:
        features_df = pd.read_csv(config.FEATURES_FILE)
        logger.info(f"Loaded {len(features_df)} features")
    except FileNotFoundError:
        logger.error("Features file not found. Run compute_features.py first.")
        return
    
    # Process a sample from each symbol (up to 3 samples per symbol)
    sample_dfs = []
    for symbol in features_df['symbol'].unique():
        symbol_data = features_df[features_df['symbol'] == symbol].head(3)  # 3 samples per symbol
        sample_dfs.append(symbol_data)
    
    sample_df = pd.concat(sample_dfs, ignore_index=True)
    results = agent.process_features(sample_df)
    
    # Display results
    print("\n" + "="*60)
    print("SIMPLE ANALYSIS RESULTS")
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
    
    print(f"\n✅ Simple analysis test completed. Processed {len(results)} samples.")


if __name__ == "__main__":
    main()
