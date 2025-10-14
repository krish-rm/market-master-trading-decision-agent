"""
News fetching module for RAG implementation.
Fetches market news and creates vector embeddings for retrieval.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logger.warning("NewsAPI not available. Install with: pip install newsapi-python")

# Make sentence transformers optional to avoid import issues
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Transformers not available: {e}. Will use mock embeddings.")


class NewsRetriever:
    """News retrieval system with vector embeddings for RAG."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news retriever.
        
        Args:
            api_key: NewsAPI key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        self.model = None
        self.index = None
        self.news_data = []
        
        if NEWSAPI_AVAILABLE and self.api_key:
            self.news_api = NewsApiClient(api_key=self.api_key)
            logger.info("✓ NewsAPI client initialized")
        else:
            self.news_api = None
            logger.warning("NewsAPI not available. Will use mock data.")
            
        # Initialize sentence transformer for embeddings
        if TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Sentence transformer model loaded")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                self.model = None
        else:
            self.model = None
            logger.info("Using mock embeddings (transformers not available)")
    
    def fetch_market_news(self, symbols: List[str], hours_back: int = 24) -> List[Dict]:
        """
        Fetch recent market news for given symbols.
        
        Args:
            symbols: List of stock symbols
            hours_back: How many hours back to fetch news
            
        Returns:
            List of news articles
        """
        if not self.news_api:
            return self._get_mock_news(symbols)
        
        news_articles = []
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)
        
        for symbol in symbols:
            try:
                # Fetch news for each symbol
                articles = self.news_api.get_everything(
                    q=f'{symbol} stock market',
                    from_param=start_date.isoformat(),
                    to=end_date.isoformat(),
                    language='en',
                    sort_by='publishedAt',
                    page_size=10
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
                                'source': article['source']['name']
                            })
                
                logger.info(f"Fetched {len(articles.get('articles', []))} articles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
        
        logger.info(f"Total news articles fetched: {len(news_articles)}")
        return news_articles
    
    def _get_mock_news(self, symbols: List[str]) -> List[Dict]:
        """Generate mock news data when NewsAPI is not available."""
        mock_news = [
            {
                'symbol': 'SPY',
                'title': 'S&P 500 Shows Strong Momentum as Tech Stocks Rally',
                'description': 'The S&P 500 index continues its upward trajectory with technology stocks leading the gains. Market analysts point to strong earnings reports and positive economic indicators.',
                'url': 'https://example.com/spy-news-1',
                'published_at': datetime.now().isoformat(),
                'source': 'Mock Financial News'
            },
            {
                'symbol': 'QQQ',
                'title': 'NASDAQ ETF QQQ Benefits from AI and Cloud Computing Trends',
                'description': 'The Invesco QQQ Trust ETF is seeing increased investor interest as artificial intelligence and cloud computing companies report strong quarterly results.',
                'url': 'https://example.com/qqq-news-1',
                'published_at': datetime.now().isoformat(),
                'source': 'Mock Tech News'
            },
            {
                'symbol': 'AAPL',
                'title': 'Apple Inc. Stock Rises on Strong iPhone Sales Reports',
                'description': 'Apple shares gained momentum following reports of stronger-than-expected iPhone sales and positive analyst upgrades citing the company\'s robust ecosystem.',
                'url': 'https://example.com/aapl-news-1',
                'published_at': datetime.now().isoformat(),
                'source': 'Mock Apple News'
            }
        ]
        
        # Filter by requested symbols
        filtered_news = [article for article in mock_news if article['symbol'] in symbols]
        logger.info(f"Generated {len(filtered_news)} mock news articles")
        return filtered_news
    
    def create_embeddings(self, news_articles: List[Dict]) -> np.ndarray:
        """
        Create embeddings for news articles.
        
        Args:
            news_articles: List of news articles
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            logger.warning("Sentence transformer not available. Using random embeddings.")
            return np.random.rand(len(news_articles), 384)
        
        # Combine title and description for embedding
        texts = []
        for article in news_articles:
            combined_text = f"{article['title']} {article['description']}"
            texts.append(combined_text)
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        logger.info(f"Created embeddings for {len(news_articles)} articles")
        
        return embeddings
    
    def build_index(self, news_articles: List[Dict], embeddings: np.ndarray):
        """
        Build FAISS index for efficient similarity search.
        
        Args:
            news_articles: List of news articles
            embeddings: Embeddings array
        """
        self.news_data = news_articles
        
        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings available. Skipping index creation.")
            return
        
        # Create FAISS index
        if TRANSFORMERS_AVAILABLE:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings.astype('float32'))
        else:
            # Mock index for when transformers not available
            self.index = None
        
        if self.index:
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        else:
            logger.info("Built mock index (transformers not available)")
    
    def retrieve_relevant_news(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant news articles for a given query.
        
        Args:
            query: Search query
            k: Number of articles to retrieve
            
        Returns:
            List of relevant news articles with similarity scores
        """
        if not self.index or not self.model:
            logger.warning("Index or model not available. Returning mock results.")
            # Return mock results for demonstration
            return [
                {
                    'title': f'Mock news for {query[:20]}...',
                    'description': 'This is mock news data for demonstration purposes.',
                    'source': 'Mock Source',
                    'similarity_score': 0.75
                }
            ]
        
        # Create embedding for query
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return relevant articles with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.news_data):
                article = self.news_data[idx].copy()
                article['similarity_score'] = float(score)
                results.append(article)
        
        logger.info(f"Retrieved {len(results)} relevant articles for query: '{query[:50]}...'")
        return results
    
    def save_news_data(self, filepath: str):
        """Save news data and embeddings to file."""
        data = {
            'news_articles': self.news_data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved news data to {filepath}")
    
    def load_news_data(self, filepath: str) -> bool:
        """Load news data from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.news_data = data['news_articles']
            logger.info(f"Loaded {len(self.news_data)} news articles from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load news data: {e}")
            return False


def main():
    """Test the news retriever."""
    logger.info("Testing News Retriever...")
    
    # Initialize retriever
    retriever = NewsRetriever()
    
    # Fetch news for our symbols
    symbols = ['SPY', 'QQQ', 'AAPL']
    news_articles = retriever.fetch_market_news(symbols, hours_back=48)
    
    if not news_articles:
        logger.warning("No news articles fetched")
        return
    
    # Create embeddings
    embeddings = retriever.create_embeddings(news_articles)
    
    # Build index
    retriever.build_index(news_articles, embeddings)
    
    # Test retrieval
    test_queries = [
        "stock market rally",
        "technology earnings",
        "market volatility"
    ]
    
    for query in test_queries:
        results = retriever.retrieve_relevant_news(query, k=2)
        print(f"\nQuery: {query}")
        for result in results:
            print(f"  - {result['symbol']}: {result['title'][:80]}... (Score: {result['similarity_score']:.3f})")
    
    # Save data
    retriever.save_news_data('data/news_data.json')
    
    print(f"\n✅ News retrieval test completed. Fetched {len(news_articles)} articles.")


if __name__ == "__main__":
    main()
