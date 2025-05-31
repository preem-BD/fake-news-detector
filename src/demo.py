#!/usr/bin/env python3
"""
Demo Script for Fake News Detector

This demo script showcases the core functionality of the Fake News Detector
without requiring API keys. It uses sample data to demonstrate:
- Clickbait detection
- Sentiment analysis
- Statistical analysis
- Data visualization

Usage:
    python src/demo.py

Authors:
    Group 4 — Kai Bleuel, Mustafa Sivgin, César Diaz Murga
    Scientific Programming WIN (2025-FS) - ZHAW
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from llm_helper import detect_clickbait_in_title, analyze_article_with_local_llm
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

def create_sample_data():
    """Create sample news articles for demonstration."""
    return [
        {
            "title": "You Won't Believe What This Celebrity Did Next!",
            "source": "ClickbaitNews",
            "content": "This shocking revelation will change everything you thought you knew about celebrities.",
            "topic": "celebrity"
        },
        {
            "title": "Federal Reserve Announces New Interest Rate Policy",
            "source": "Financial Times", 
            "content": "The Federal Reserve has announced a new monetary policy framework targeting inflation control.",
            "topic": "economy"
        },
        {
            "title": "Scientists Discover New Species in Amazon Rainforest",
            "source": "Nature Journal",
            "content": "Researchers have identified a previously unknown primate species in the Amazon basin.",
            "topic": "science"
        },
        {
            "title": "This Simple Trick Will Double Your Income Overnight!",
            "source": "MoneyHacks",
            "content": "Financial experts hate this one weird trick that can transform your finances instantly.",
            "topic": "finance"
        },
        {
            "title": "Climate Change Report Shows Alarming Temperature Trends",
            "source": "Environmental Science",
            "content": "Latest IPCC report reveals concerning data about global temperature increases over the past decade.",
            "topic": "climate"
        }
    ]

def analyze_sample_articles():
    """Analyze sample articles and display results."""
    print(f"{Fore.CYAN}🎭 FAKE NEWS DETECTOR - DEMO MODE{Style.RESET_ALL}")
    print("=" * 50)
    print(f"{Fore.YELLOW}📝 Analyzing sample articles (no API keys required)...{Style.RESET_ALL}\n")
    
    articles = create_sample_data()
    results = []
    
    for i, article in enumerate(articles, 1):
        print(f"{Fore.BLUE}📰 Article {i}: {article['title'][:60]}...{Style.RESET_ALL}")
        print(f"   Source: {article['source']}")
        
        # Clickbait detection
        clickbait_result = detect_clickbait_in_title(article['title'])
        clickbait_clean = "Clickbait" if "❌" in clickbait_result else "Not Clickbait"
        
        # Sentiment analysis
        sentiment_result = analyze_article_with_local_llm(article['content'])
        if "NEGATIVE" in sentiment_result.upper():
            sentiment_clean = "Negative"
        elif "POSITIVE" in sentiment_result.upper():
            sentiment_clean = "Positive"
        else:
            sentiment_clean = "Neutral"
        
        print(f"   {clickbait_result}")
        print(f"   {sentiment_result}")
        print()
        
        results.append({
            'Article': i,
            'Title': article['title'],
            'Source': article['source'],
            'Topic': article['topic'],
            'Clickbait': clickbait_clean,
            'Sentiment': sentiment_clean,
            'Word_Count': len(article['content'].split())
        })
    
    return results

def create_visualizations(results):
    """Create sample visualizations from the analysis results."""
    df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Clickbait distribution pie chart
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    clickbait_counts = df['Clickbait'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    plt.pie(clickbait_counts.values, labels=clickbait_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Clickbait Detection Results')
    
    # Sentiment distribution
    plt.subplot(1, 2, 2)
    sentiment_counts = df['Sentiment'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Sentiment Analysis Results')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'demo_analysis_charts.png', dpi=300, bbox_inches='tight')
    print(f"{Fore.GREEN}📊 Charts saved to: demo_output/demo_analysis_charts.png{Style.RESET_ALL}")
    
    # Save results to CSV
    df.to_csv(output_dir / 'demo_results.csv', index=False)
    print(f"{Fore.GREEN}💾 Results saved to: demo_output/demo_results.csv{Style.RESET_ALL}")
    
    return df

def display_statistics(df):
    """Display statistical analysis of the demo results."""
    print(f"\n{Fore.CYAN}📊 STATISTICAL ANALYSIS{Style.RESET_ALL}")
    print("=" * 30)
    
    # Basic statistics
    print(f"📈 Total articles analyzed: {len(df)}")
    print(f"🎯 Clickbait detected: {len(df[df['Clickbait'] == 'Clickbait'])} ({len(df[df['Clickbait'] == 'Clickbait'])/len(df)*100:.1f}%)")
    print(f"😊 Positive sentiment: {len(df[df['Sentiment'] == 'Positive'])} ({len(df[df['Sentiment'] == 'Positive'])/len(df)*100:.1f}%)")
    print(f"😐 Neutral sentiment: {len(df[df['Sentiment'] == 'Neutral'])} ({len(df[df['Sentiment'] == 'Neutral'])/len(df)*100:.1f}%)")
    print(f"😞 Negative sentiment: {len(df[df['Sentiment'] == 'Negative'])} ({len(df[df['Sentiment'] == 'Negative'])/len(df)*100:.1f}%)")
    print(f"📝 Average word count: {df['Word_Count'].mean():.1f}")
    
    # Cross-tabulation
    print(f"\n{Fore.YELLOW}🔗 Clickbait vs Sentiment Correlation:{Style.RESET_ALL}")
    crosstab = pd.crosstab(df['Clickbait'], df['Sentiment'])
    print(crosstab)

def main():
    """Main demo function."""
    try:
        print(f"{Fore.MAGENTA}🚀 Starting Fake News Detector Demo...{Style.RESET_ALL}\n")
        
        # Analyze sample articles
        results = analyze_sample_articles()
        
        # Create visualizations
        df = create_visualizations(results)
        
        # Display statistics
        display_statistics(df)
        
        print(f"\n{Fore.GREEN}✅ Demo completed successfully!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎯 Next steps:{Style.RESET_ALL}")
        print("   1. Get a free NewsAPI key from https://newsapi.org/")
        print("   2. Create .env file with: NEWS_API_KEY=your_key_here")
        print("   3. Run: python src/main.py (for terminal interface)")
        print("   4. Run: streamlit run src/streamlit_app.py (for web interface)")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Demo error: {e}{Style.RESET_ALL}")
        print("This is normal if some dependencies are missing in demo mode.")

if __name__ == "__main__":
    main()
