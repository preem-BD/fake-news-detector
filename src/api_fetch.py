"""
NewsAPI Integration Module

This module handles fetching real-world news articles from NewsAPI.org.
It provides functions to search for articles based on keywords and retrieve
structured data for analysis.

Dependencies:
    - requests: For HTTP API calls
    - python-dotenv: For loading API keys from .env file
    - random: For randomizing search results across pages

Environment Variables Required:
    NEWS_API_KEY: Your NewsAPI.org API key (stored in .env file)
"""

import requests
import os
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

def fetch_articles(query="politics", page_size=5, sort_by="publishedAt"):
    """
    Fetch news articles from NewsAPI based on search query.
    
    This function makes an HTTP request to NewsAPI.org to retrieve articles
    matching the specified query. It randomly selects a page (1-5) to get
    varied results and avoid always seeing the same articles.
    
    Args:
        query (str): Search keyword/phrase (default: "politics")
        page_size (int): Number of articles to fetch (default: 5, max: 100)
        sort_by (str): Sort criteria - "publishedAt", "relevancy", or "popularity"
                      (default: "publishedAt")
    
    Returns:
        list: List of article dictionaries containing:
              - title: Article headline
              - description: Article summary
              - url: Link to full article
              - source: Dictionary with news source info
              - publishedAt: Publication timestamp
              - content: Partial article content
    
    Raises:
        ValueError: If API returns an error (invalid key, quota exceeded, etc.)
        
    Example:
        >>> articles = fetch_articles("bitcoin", page_size=10)
        >>> print(f"Found {len(articles)} articles about bitcoin")
    """    # Generate random page number (1-5) to get varied results
    # This prevents always seeing the same articles for popular queries
    random_page = random.randint(1, 5)

    # Construct NewsAPI URL with query parameters
    # API Documentation: https://newsapi.org/docs/endpoints/everything
    url = (
        f"https://newsapi.org/v2/everything?"      # Base API endpoint
        f"q={query}"                               # Search query
        f"&pageSize={page_size}"                   # Number of articles per page
        f"&page={random_page}"                     # Page number (randomized)
        f"&sortBy={sort_by}"                       # Sort order
        f"&language=en"                            # English articles only
        f"&apiKey={API_KEY}"                       # Authentication
    )

    # Make HTTP GET request to NewsAPI
    response = requests.get(url)
    data = response.json()

    # Error handling - check if API returned valid response
    if "articles" not in data:
        # Common errors: invalid API key, quota exceeded, malformed query
        raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")

    # Success feedback with page info
    print(f"✅ Fetched page {random_page} with {len(data['articles'])} articles.")
    
    # Return the articles array from API response
    return data["articles"]
