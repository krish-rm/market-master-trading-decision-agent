"""
LLM Agent module for LLM Market Decision Agent.
Uses Groq API for fast, free AI-powered market analysis.
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


class MarketDecisionAgent:
    """
    Groq-powered market decision agent that analyzes technical indicators
    and provides trading guidance using fast, free AI.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        prompt_template_path: Path = None
    ):
        """
        Initialize the Market Decision Agent with Groq.
        
        Args:
            api_key: Groq API key (if None, uses config.GROQ_API_KEY)
            model: Groq model name (if None, uses config.GROQ_MODEL)
            prompt_template_path: Path to prompt template file
        """
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self.provider = "groq"
        
        # Initialize Groq client
        if self.api_key and self.api_key != "":
            self.client = Groq(api_key=self.api_key)
            self.use_api = True
            logger.info(f"✓ Initialized Groq with model: {self.model}")
        else:
            self.client = None
            self.use_api = False
            logger.warning("No Groq API key provided. Will use fallback mode.")
            logger.warning("Get free API key at: https://console.groq.com/")
        
        # Load prompt template
        if prompt_template_path is None:
            prompt_template_path = config.PROMPTS_DIR / "decision_prompt.txt"
        
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        
        logger.info("Market Decision Agent initialized")
    
    def _format_prompt(self, row: pd.Series) -> str:
        """
        Format the prompt with market data.
        
        Args:
            row: DataFrame row with market indicators
        
        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(
            symbol=row['symbol'],
            timestamp=row['timestamp'],
            close_price=row['close'],
            rsi=row['rsi'],
            atr=row['atr'],
            volume_bias=row['volume_bias'],
            trend=row['trend'],
            wss=row['wss']
        )
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Parse JSON response from LLM, handling various formats.
        
        Args:
            response_text: Raw response text from LLM
        
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try to find JSON block in markdown code fence
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                # Try to find JSON object directly
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end].strip()
            
            return json.loads(json_str)
        
        except Exception as e:
            logger.debug(f"JSON parse error: {str(e)}")  # Changed to debug level
            # Return default structure (silently handle parsing errors)
            return {
                "reasoning": "Failed to parse LLM response",
                "guidance": "Unable to provide guidance",
                "confidence": "Low"
            }
    
    def _generate_fallback_response(self, row: pd.Series) -> Dict:
        """
        Generate a rule-based fallback response when API is unavailable.
        
        Args:
            row: DataFrame row with market indicators
        
        Returns:
            Dictionary with reasoning, guidance, and confidence
        """
        rsi = row['rsi']
        wss = row['wss']
        trend = row['trend']
        volume_bias = row['volume_bias']
        
        # Simple rule-based logic
        if wss > 0.7 and trend == 'up' and rsi < 70:
            reasoning = f"Strong bullish sentiment (WSS: {wss:.2f}) with uptrend and RSI at {rsi:.1f}. Volume is {volume_bias:.2f}x average, indicating strong participation."
            guidance = "Consider long positions with tight stops. Monitor for overbought conditions as RSI approaches 70."
            confidence = "High"
        elif wss < 0.3 and trend == 'down' and rsi > 30:
            reasoning = f"Weak sentiment (WSS: {wss:.2f}) with downtrend and RSI at {rsi:.1f}. Volume bias of {volume_bias:.2f}x suggests selling pressure."
            guidance = "Consider short positions or reduce longs. Watch for oversold bounce if RSI drops below 30."
            confidence = "High"
        elif 0.4 <= wss <= 0.6:
            reasoning = f"Neutral sentiment (WSS: {wss:.2f}) with {trend} trend and RSI at {rsi:.1f}. Mixed signals suggest consolidation."
            guidance = "Stay neutral or reduce position sizes. Wait for clearer directional signals."
            confidence = "Low"
        else:
            reasoning = f"Mixed signals: WSS {wss:.2f}, {trend} trend, RSI {rsi:.1f}. Indicators not fully aligned."
            guidance = "Exercise caution. Consider smaller positions or wait for better setup."
            confidence = "Medium"
        
        return {
            "reasoning": reasoning,
            "guidance": guidance,
            "confidence": confidence
        }
    
    def analyze_market(self, row: pd.Series, max_retries: int = 3) -> Tuple[Dict, bool]:
        """
        Analyze market conditions and generate trading guidance.
        
        Args:
            row: DataFrame row with market indicators
            max_retries: Maximum number of API retry attempts
        
        Returns:
            Tuple of (response_dict, used_api)
        """
        # Use fallback if API not available
        if not self.use_api:
            response = self._generate_fallback_response(row)
            return response, False
        
        # Format prompt
        prompt = self._format_prompt(row)
        
        # Try API call with retries
        for attempt in range(max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
                
                # Call Groq API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert market strategist. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE
                )
                response_text = response.choices[0].message.content
                
                # Parse response
                parsed_response = self._parse_json_response(response_text)
                
                return parsed_response, True
            
            except Exception as e:
                error_str = str(e)
                logger.warning(f"API call failed (attempt {attempt + 1}): {error_str}")
                
                # Check for specific rate limit errors
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    if "requests per day" in error_str:
                        logger.error("Daily rate limit reached. Using fallback for remaining requests.")
                        response = self._generate_fallback_response(row)
                        return response, False
                    elif "requests per min" in error_str:
                        # Wait longer for per-minute rate limits
                        wait_time = 30 + (attempt * 10)  # 30s, 40s, 50s
                        logger.warning(f"Per-minute rate limit hit. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Max retries reached. Using fallback.")
                    response = self._generate_fallback_response(row)
                    return response, False
        
        # Should not reach here, but just in case
        response = self._generate_fallback_response(row)
        return response, False


def process_features(
    features_df: pd.DataFrame,
    agent: MarketDecisionAgent,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Process all features through the LLM agent.
    
    Args:
        features_df: DataFrame with computed features
        agent: MarketDecisionAgent instance
        sample_size: If provided, process only this many random rows
    
    Returns:
        DataFrame with LLM outputs
    """
    logger.info("Processing features through LLM agent...")
    
    # Always process the latest N rows per symbol (default: 20)
    latest_n = 20 if sample_size is None else sample_size
    logger.info(f"Selecting latest {latest_n} rows per symbol for processing")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(features_df['timestamp']):
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

    # Sort by timestamp descending per symbol and take the first N (latest)
    features_df_sorted = features_df.sort_values(['symbol', 'timestamp'], ascending=[True, False])
    df_to_process = (
        features_df_sorted.groupby('symbol', as_index=False, group_keys=False)
        .apply(lambda g: g.head(latest_n))
        .reset_index(drop=True)
    )
    
    results = []
    total = len(df_to_process)
    api_calls = 0
    fallback_calls = 0
    
    for idx, row in df_to_process.iterrows():
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{total} rows...")
        
        # Get LLM analysis
        analysis, used_api = agent.analyze_market(row)
        
        if used_api:
            api_calls += 1
        else:
            fallback_calls += 1
        
        # Combine row data with analysis
        result = {
            'timestamp': row['timestamp'],
            'symbol': row['symbol'],
            'close': row['close'],
            'rsi': row['rsi'],
            'atr': row['atr'],
            'volume_bias': row['volume_bias'],
            'trend': row['trend'],
            'wss': row['wss'],
            'reasoning': analysis['reasoning'],
            'guidance': analysis['guidance'],
            'confidence': analysis['confidence']
        }
        
        results.append(result)
        
        # Rate limiting for API calls
        if used_api:
            time.sleep(1.0)  # Increased delay to respect rate limits
    
    logger.info(f"Processing complete: {api_calls} API calls, {fallback_calls} fallback calls")
    
    return pd.DataFrame(results)


def main():
    """Main execution function."""
    try:
        # Initialize Groq-powered agent
        agent = MarketDecisionAgent()
        
        # Load features
        logger.info(f"Loading features from {config.FEATURES_FILE}")
        features_df = pd.read_csv(config.FEATURES_FILE)
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # Process features with balanced sample size for comprehensive analysis
        # This gives good coverage across all symbols without excessive API calls
        outputs_df = process_features(
            features_df,
            agent,
            sample_size=30  # 30 samples for comprehensive 1-hour analysis
        )
        
        # Save outputs
        outputs_df.to_csv(config.LLM_OUTPUTS_FILE, index=False)
        logger.info(f"LLM outputs saved to {config.LLM_OUTPUTS_FILE}")
        
        # Display summary
        print("\n" + "="*60)
        print("LLM AGENT PROCESSING SUMMARY")
        print("="*60)
        print(f"Provider: {agent.provider.upper()}")
        print(f"Model: {agent.model}")
        print(f"Total rows processed: {len(outputs_df)}")
        print(f"API mode: {'Enabled' if agent.use_api else 'Fallback (no API key)'}")
        print(f"\nConfidence distribution:")
        print(outputs_df['confidence'].value_counts())
        print(f"\nOutputs saved to: {config.LLM_OUTPUTS_FILE}")
        print("="*60 + "\n")
        
        # Display sample outputs
        print("Sample LLM Outputs:")
        for _, row in outputs_df.head(3).iterrows():
            print(f"\n{row['symbol']} @ {row['timestamp']}")
            print(f"WSS: {row['wss']:.2f} | Trend: {row['trend']} | Confidence: {row['confidence']}")
            
            # Safely handle string slicing
            reasoning = str(row['reasoning']) if row['reasoning'] is not None else "No reasoning provided"
            guidance = str(row['guidance']) if row['guidance'] is not None else "No guidance provided"
            
            print(f"Reasoning: {reasoning[:150]}...")
            print(f"Guidance: {guidance[:150]}...")
            print("-" * 60)
        
    except Exception as e:
        logger.error(f"Failed to process LLM agent: {str(e)}")
        raise


if __name__ == "__main__":
    main()

