"""
Simple news fetcher without sentence transformers dependencies.
Provides mock news data for RAG demonstration.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logger.warning("NewsAPI not available. Will use mock data.")


class SimpleNewsRetriever:
    """Simple news retrieval system without transformer dependencies."""
    
    def __init__(self, api_key: str = None):
        """Initialize simple news retriever."""
        self.api_key = api_key or config.NEWSAPI_KEY
        
        if NEWSAPI_AVAILABLE and self.api_key:
            self.news_api = NewsApiClient(api_key=self.api_key)
            logger.info("✓ NewsAPI client initialized")
        else:
            self.news_api = None
            logger.warning("NewsAPI not available. Will use mock data.")
    
    def fetch_market_news(self, symbols: List[str], hours_back: int = 24) -> List[Dict]:
        """Fetch market news or generate mock data."""
        if self.news_api:
            return self._fetch_real_news(symbols, hours_back)
        else:
            return self._get_mock_news(symbols)
    
    def _fetch_real_news(self, symbols: List[str], hours_back: int) -> List[Dict]:
        """Fetch real news from NewsAPI."""
        news_articles = []
        
        for symbol in symbols:
            try:
                # Simple news fetch without complex dependencies
                articles = self.news_api.get_everything(
                    q=f'{symbol} stock market',
                    language='en',
                    sort_by='publishedAt',
                    page_size=5
                )
                
                if articles['status'] == 'ok':
                    for article in articles['articles']:
                        if article['title'] and article['description']:
                            news_articles.append({
                                'symbol': symbol,
                                'title': article['title'],
                                'description': article['description'],
                                'url': article['url'],
                                'published_at': article['publishedAt'],
                                'source': article['source']['name'],
                                'similarity_score': 0.75  # Mock similarity score
                            })
                
                logger.info(f"Fetched {len(articles.get('articles', []))} articles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
        
        return news_articles
    
    def _get_mock_news(self, symbols: List[str]) -> List[Dict]:
        """Generate mock news data."""
        mock_news = []
        
        for symbol in symbols:
            mock_news.extend([
                {
                    'symbol': symbol,
                    'title': f'{symbol} Shows Strong Market Performance',
                    'description': f'{symbol} demonstrates robust trading activity with positive momentum indicators.',
                    'url': f'https://example.com/{symbol.lower()}-news-1',
                    'published_at': datetime.now().isoformat(),
                    'source': 'Mock Financial News',
                    'similarity_score': 0.80
                },
                {
                    'symbol': symbol,
                    'title': f'Analysts Bullish on {symbol} Outlook',
                    'description': f'Market analysts maintain positive outlook for {symbol} based on technical indicators.',
                    'url': f'https://example.com/{symbol.lower()}-news-2',
                    'published_at': datetime.now().isoformat(),
                    'source': 'Mock Analysis Report',
                    'similarity_score': 0.70
                },
                {
                    'symbol': symbol,
                    'title': f'{symbol} Trading Volume Above Average',
                    'description': f'{symbol} experiences elevated trading volume indicating strong market interest.',
                    'url': f'https://example.com/{symbol.lower()}-news-3',
                    'published_at': datetime.now().isoformat(),
                    'source': 'Mock Market Data',
                    'similarity_score': 0.65
                }
            ])
        
        logger.info(f"Generated {len(mock_news)} mock news articles")
        return mock_news
    
    def retrieve_relevant_news(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant news based on query (simplified version)."""
        # For simplicity, just return mock news
        # In a real implementation, you'd use text matching or simple keyword search
        return [
            {
                'title': f'Market Analysis for {query[:20]}...',
                'description': f'Current market conditions show strong activity related to {query}.',
                'source': 'Mock Analysis',
                'similarity_score': 0.75
            }
        ]
    
    def save_news_data(self, filepath: str):
        """Save news data to file."""
        # Generate mock news data
        symbols = ['SPY', 'QQQ', 'AAPL']
        news_articles = self.fetch_market_news(symbols)
        
        data = {
            'news_articles': news_articles,
            'timestamp': datetime.now().isoformat(),
            'source': 'simple_retriever'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved news data to {filepath}")


def main():
    """Test the simple news retriever."""
    logger.info("Testing Simple News Retriever...")
    
    retriever = SimpleNewsRetriever()
    
    # Test news fetching
    symbols = ['SPY', 'QQQ', 'AAPL']
    news_articles = retriever.fetch_market_news(symbols)
    
    print(f"\n✅ Fetched {len(news_articles)} news articles")
    
    # Test news retrieval
    test_query = "stock market rally"
    relevant_news = retriever.retrieve_relevant_news(test_query)
    
    print(f"✅ Retrieved {len(relevant_news)} relevant articles for query: '{test_query}'")
    
    # Save data
    retriever.save_news_data(str(config.NEWS_DATA_FILE))
    
    print(f"✅ Simple news retrieval test completed.")


if __name__ == "__main__":
    main()
