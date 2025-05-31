"""
Data Cleaning and Processing Module

This module provides functions for cleaning article text and converting 
raw API responses into structured pandas DataFrames.

Functions:
    clean_text(text): Cleans and normalizes text data
    articles_to_df(articles): Converts article list to pandas DataFrame
"""

import re
import pandas as pd

def clean_text(text):
    """
    Clean and normalize text data by removing URLs, punctuation, and extra spaces.
    
    This function performs several text preprocessing steps:
    1. Removes all URLs (http/https links)
    2. Removes all punctuation marks
    3. Normalizes whitespace (multiple spaces become single space)
    4. Converts text to lowercase
    
    Args:
        text (str): Raw text to be cleaned
        
    Returns:
        str: Cleaned and normalized text
        
    Example:
        >>> clean_text("Check this out! https://example.com   Amazing stuff!!!")
        "check this out amazing stuff"
    """
    # Remove URLs (anything starting with http)
    text = re.sub(r"http\S+", "", text)  
    
    # Remove all punctuation (keep only word characters and spaces)
    text = re.sub(r"[^\w\s]", "", text)  
    
    # Replace multiple whitespace characters with single space
    text = re.sub(r"\s+", " ", text)     
    
    # Convert to lowercase for consistency
    return text.lower()

def articles_to_df(articles):
    """
    Convert a list of article dictionaries into a pandas DataFrame.
    
    This function processes raw article data from NewsAPI and creates a structured
    DataFrame with cleaned text for analysis. It combines title and description
    for more comprehensive text analysis.
    
    Args:
        articles (list): List of article dictionaries from NewsAPI
                        Each article should have: title, description, source, publishedAt
        
    Returns:
        pd.DataFrame: DataFrame with columns:
                     - source: News source name
                     - title: Original article title
                     - text: Cleaned combined title + description
                     - publishedAt: Publication timestamp
                     
    Example:
        >>> articles = [{"title": "News Title", "description": "News desc", 
                        "source": {"name": "CNN"}, "publishedAt": "2025-01-01"}]
        >>> df = articles_to_df(articles)
        >>> df.columns
        ['source', 'title', 'text', 'publishedAt']
    """
    records = []
    
    # Process each article from the API response
    for art in articles:
        # Combine title and description, handle missing description
        title = art["title"]
        description = art.get("description") or ""  # Use empty string if None
        
        # Clean the combined text for analysis
        cleaned = clean_text(title + " " + description)
        
        # Create structured record
        records.append({
            "source": art["source"]["name"],     # News source name
            "title": art["title"],               # Keep original title
            "text": cleaned,                     # Cleaned text for analysis
            "publishedAt": art["publishedAt"]    # Publication timestamp
        })
    
    # Convert list of records to pandas DataFrame
    return pd.DataFrame(records)
