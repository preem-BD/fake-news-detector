"""
Data Analysis and Visualization Module

This module provides statistical analysis and visualization functions for
article data analysis. It includes functions for plotting distributions
and comparing text lengths between different groups.

Dependencies:
    - pandas: Data manipulation
    - matplotlib: Basic plotting
    - seaborn: Statistical visualizations
    - scipy: Statistical tests
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def plot_word_counts(df):
    """
    Create a histogram showing the distribution of word counts in articles.
    
    This function adds a 'word_count' column to the DataFrame and creates
    a histogram visualization to show how article lengths are distributed.
    
    Args:
        df (pd.DataFrame): DataFrame with 'text' column containing article text
        
    Returns:
        None: Displays the plot using matplotlib
        
    Example:
        >>> df = pd.DataFrame({'text': ['short text', 'much longer text here']})
        >>> plot_word_counts(df)
        # Displays histogram
    """
    # Calculate word count for each article by splitting on whitespace
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    
    # Create histogram with 10 bins
    sns.histplot(df["word_count"], bins=10)
    plt.title("Distribution of Article Word Counts")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.show()

def compare_lengths(df1, df2):
    """
    Compare the average text lengths between two groups using a t-test.
    
    This function performs an independent samples t-test to determine if
    there is a statistically significant difference in word counts between
    two groups of articles.
    
    Args:
        df1 (pd.DataFrame): First group with 'text' column
        df2 (pd.DataFrame): Second group with 'text' column
        
    Returns:
        tuple: (t-statistic, p-value)
               - t-statistic: The calculated t-statistic
               - p-value: The two-tailed p-value
               
    Example:
        >>> clickbait_articles = df[df['clickbait'] == 'Clickbait']
        >>> normal_articles = df[df['clickbait'] == 'Not Clickbait']
        >>> t_stat, p_val = compare_lengths(clickbait_articles, normal_articles)
        >>> print(f"Difference significant: {p_val < 0.05}")
    """
    # Calculate word counts for both groups
    group1_lengths = df1["text"].apply(lambda x: len(x.split()))
    group2_lengths = df2["text"].apply(lambda x: len(x.split()))
    
    # Perform independent samples t-test
    stat, p = ttest_ind(group1_lengths, group2_lengths)
    
    # Display results
    print(f"T-test result: t={stat:.2f}, p={p:.4f}")
    
    return stat, p
