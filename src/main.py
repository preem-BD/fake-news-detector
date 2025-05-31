"""
Main Terminal Application for Fake News Detection

This is the primary command-line interface for the Fake News Detector project.
It orchestrates the entire analysis pipeline from data collection to statistical
analysis and visualization.

Features:
    - Interactive keyword-based article search
    - Real-time sentiment analysis using AI
    - Clickbait detection using pattern matching
    - Statistical correlation analysis with p-values
    - Data visualization with pie charts
    - SQLite database integration
    - Colored terminal output for better UX

Usage:
    python main.py

Authors:
    Group 4 — Kai Bleuel, Mustafa Sivgin, César Diaz Murga
    Scientific Programming WIN (2025-FS) - ZHAW
"""

# Import our custom modules
from api_fetch import fetch_articles
from clean_data import articles_to_df
from llm_helper import analyze_article_with_local_llm, detect_clickbait_in_title

# Import standard libraries
import sqlite3  # For database operations
import pandas as pd  # For data manipulation
import os  # For file operations
import matplotlib.pyplot as plt  # For plotting
from scipy.stats import pearsonr, spearmanr, chi2_contingency  # For statistics
from colorama import Fore, Style, init  # For colored terminal output

# Initialize colorama for cross-platform colored output
init(autoreset=True)

def highlight_keyword(text, keyword, base_color=Fore.WHITE):
    """
    Highlight keyword occurrences in text using terminal colors.
    
    This function searches for keyword matches in text and applies red coloring
    to make them visually stand out in the terminal output.
    
    Args:
        text (str): The text to search and highlight in
        keyword (str): The keyword to highlight (case-insensitive)
        base_color (str): Base color for non-highlighted text
        
    Returns:
        str: Text with ANSI color codes for highlighted keywords
        
    Example:
        >>> highlight_keyword("Apple iPhone news", "iphone")
        "Apple iPhone news"  # with "iPhone" in red
    """
    keyword_lower = keyword.lower()
    highlighted_text = ""
    i = 0
    
    # Iterate through text character by character
    while i < len(text):
        # Check if current position matches keyword (case-insensitive)
        if text[i:i+len(keyword)].lower() == keyword_lower:
            # Add red highlighting for keyword
            highlighted_text += Fore.RED + text[i:i+len(keyword)] + base_color
            i += len(keyword)
        else:
            # Add regular character
            highlighted_text += text[i]
            i += 1
    
    return highlighted_text

def run_statistical_analysis(df_results):
    """
    Perform comprehensive statistical analysis on article data.
    
    This function conducts three types of statistical tests to explore
    relationships between different article characteristics:
    
    1. Pearson correlation: Linear relationship between sentiment & word count
    2. Spearman correlation: Monotonic relationship between sentiment & word count  
    3. Chi-Square test: Independence between clickbait detection & sentiment
    
    All tests provide p-values to assess statistical significance (p < 0.05).
    
    Args:
        df_results (pd.DataFrame): Results with columns:
                                  - Sentiment Score (float)
                                  - Word Count (int)
                                  - Clickbait (str)
                                  - Sentiment (str)
    
    Returns:
        None: Prints results to terminal with colored output
    """
    print(Fore.CYAN + "\n--- Statistical Analysis ---")

    # --- PART A: Sentiment Score ↔ Word Count Correlations ---
    print(Fore.CYAN + "\n📊 Sentiment Score vs. Word Count:")

    sentiment_scores = df_results["Sentiment Score"]
    word_counts = df_results["Word Count"]

    # Pearson correlation: measures linear relationship (-1 to +1)
    pearson_corr, pearson_p = pearsonr(sentiment_scores, word_counts)

    # Spearman correlation: measures monotonic relationship (rank-based)
    spearman_corr, spearman_p = spearmanr(sentiment_scores, word_counts)

    # Display correlation results with significance indicators
    print(Fore.YELLOW + f"Pearson correlation: r = {pearson_corr:.2f}, p-value = {pearson_p:.4f}")
    print(Fore.YELLOW + f"Spearman correlation: r = {spearman_corr:.2f}, p-value = {spearman_p:.4f}")

    # --- PART B: Clickbait vs. Sentiment Independence Test ---
    print(Fore.CYAN + "\n📊 Clickbait vs. Sentiment (Chi-Square Test):")

    # Build contingency table for categorical variables
    contingency_table = pd.crosstab(df_results["Clickbait"], df_results["Sentiment"])

    # Chi-Square test: tests independence of categorical variables    # Chi-Square test: tests independence of categorical variables
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Display Chi-Square test results
    print(Fore.YELLOW + f"Chi2 = {chi2:.2f}, p-value = {p_value:.4f}, dof = {dof}")
    print(Fore.YELLOW + "\nExpected frequencies:")
    print(expected)

def main():
    """
    Main application entry point and user interface.
    
    This function orchestrates the entire analysis workflow:
    1. User input for search topic
    2. Article fetching from NewsAPI
    3. Data cleaning and filtering
    4. AI-powered analysis (sentiment + clickbait)
    5. Data visualization (pie chart)
    6. User interaction for article selection
    7. Database storage (SQLite)
    8. Statistical analysis with p-values
    
    The function provides a complete interactive experience in the terminal
    with colored output and user-friendly prompts.
    """
    # Welcome screen with project info
    print(Fore.CYAN + "📰 Welcome to Fake News Detector!")
    print(Fore.CYAN + "---------------------------------")
    print(Fore.CYAN + "Group 4 — Kai Bleuel, Mustafa Sivgin, César Diaz Murga")
    print()

    # Get search topic from user
    topic_input = input(Fore.YELLOW + "Enter a topic to search for: ").strip()
    topic = topic_input.lower()

    # Step 1: Fetch articles from NewsAPI
    print(Fore.CYAN + f"\n🚀 Searching articles for topic: '{topic}' ...")
    articles = fetch_articles(topic, page_size=20)

    # Handle case where no articles found
    if not articles:
        print(Fore.RED + f"\n❌ No articles found for topic: '{topic}'.")
        return

    # Step 2: Clean and structure data
    print(Fore.CYAN + "🧹 Cleaning and converting to DataFrame...")
    df = articles_to_df(articles)

    # Step 3: Filter articles that actually contain the keyword
    print(Fore.CYAN + "\n🔎 Filtering articles that contain your keyword...")
    df_filtered = df[
        df["title"].str.contains(topic, case=False, na=False) |
        df["text"].str.contains(topic, case=False, na=False)
    ]

    # Handle case where no relevant articles found
    if df_filtered.empty:
        print(Fore.RED + f"\n❌ No articles found that actually contain '{topic}'.")
        return

    print(Fore.GREEN + f"\n✅ Analyzing {len(df_filtered)} articles...")

    # Step 4: Analyze all articles for clickbait + sentiment
    results = []

    for idx, (i, row) in enumerate(df_filtered.iterrows(), start=1):
        # Detect clickbait characteristics in title
        clickbait_result = detect_clickbait_in_title(row.get('title', ''))
        clickbait_label = clickbait_result

        # Analyze sentiment using AI model
        sentiment_result = analyze_article_with_local_llm(row["text"])
        sentiment_label = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]        # Create combined classification label
        combined_label = f"{clickbait_label} + {sentiment_label}"

        # Calculate word count as a measure of article length
        word_count = len(row.get('text', '').split())

        # Store all analysis results in structured format
        results.append({
            "Article": f"{idx}",                     # Sequential numbering
            "Title": row.get('title', ''),           # Original title
            "Clickbait": clickbait_label,            # Clickbait classification
            "Sentiment": sentiment_label,            # Sentiment classification
            "Sentiment Score": sentiment_score,      # AI confidence score
            "Combined Label": combined_label,        # Combined classification
            "Word Count": word_count                 # Text length metric
        })

    # Step 5: Convert results to DataFrame for analysis and visualization
    df_results = pd.DataFrame(results)

    # Step 6: Create pie chart visualization
    label_counts = df_results["Combined Label"].value_counts()

    print(Fore.CYAN + "\n📊 Showing PIE chart of Clickbait + Sentiment:")

    # Create and display pie chart with customized styling
    plt.figure(figsize=(3, 3))  # Compact size for terminal display
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=plt.cm.tab20.colors)
    plt.title(f"Clickbait + Sentiment distribution for '{topic}'")
    plt.tight_layout()
    plt.show()

    # Step 7: Display article list for user selection
    print(Fore.CYAN + "\n--- Article List ---")
    for idx, row in df_results.iterrows():
        # Highlight keyword in article titles
        title_highlighted = highlight_keyword(row["Title"], topic, base_color=Fore.YELLOW)
        print(Fore.YELLOW + f"[{row['Article']}] {title_highlighted}")

    # Step 8: Interactive article selection with input validation
    while True:
        try:
            choice = int(input(f"\nEnter the number of the article you want to analyze (1–{len(df_results)}): "))
            if 1 <= choice <= len(df_results):
                break
            else:
                print(Fore.RED + "⚠️ Invalid number, try again.")
        except ValueError:
            print(Fore.RED + "⚠️ Please enter a valid integer.")

    # Get selected article data
    selected = df_results.iloc[choice - 1]
    df_full = df_filtered.iloc[choice - 1]

    # Step 9: Display detailed analysis of selected article
    print(Fore.MAGENTA + f"\n🔍 Analyzing selected article [{selected['Article']}]")
    title_highlighted = highlight_keyword(selected["Title"], topic, base_color=Fore.YELLOW)
    print(Fore.YELLOW + f"📰 Title: {title_highlighted}")    # Display full article text with keyword highlighting
    full_text = df_full.get('text') or df_full.get('content') or df_full.get('description') or df_full.get('title', '')
    text_highlighted = highlight_keyword(full_text, topic, base_color=Fore.BLUE)
    print(Fore.BLUE + f"\n📝 Full Text:\n{text_highlighted}\n")

    # Display analysis results with color-coded output
    
    # Clickbait detection result (Green = safe, Red = clickbait)
    clickbait_color = Fore.GREEN if selected['Clickbait'] == "Not Clickbait" else Fore.RED
    print(f"{clickbait_color}🚩 CLICKBAIT RESULT: {selected['Clickbait'].upper()}")

    # Sentiment analysis result with appropriate colors
    if selected['Sentiment'] == "Positive":
        sentiment_color = Fore.GREEN
    elif selected['Sentiment'] == "Negative":
        sentiment_color = Fore.RED
    else:
        sentiment_color = Fore.LIGHTBLACK_EX
    print(f"{sentiment_color}🎭 SENTIMENT RESULT: {selected['Sentiment'].upper()} (score: {selected['Sentiment Score']})")

    # Article length metric
    print(f"{sentiment_color}📝 WORD COUNT: {selected['Word Count']}")

    # Step 10: Save results to CSV file
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)  # Create directory if it doesn't exist
    output_path = os.path.join(output_folder, "combined_analysis_with_pie.csv")
    df_results.to_csv(output_path, index=False)

    print(Fore.YELLOW + f"\n✅ Combined results saved to: {output_path}")
    print(Fore.YELLOW + "👉 To run the Streamlit app, use: streamlit run src/streamlit_app.py")

    # Step 11: Database integration - Save to SQLite
    sqlite_path = os.path.join(output_folder, "fake_news_analysis.db")

    # Connect to SQLite database (creates file if doesn't exist)
    conn = sqlite3.connect(sqlite_path)

    # Save DataFrame to SQL table named "articles"
    # if_exists="replace" overwrites existing table
    df_results.to_sql("articles", conn, if_exists="replace", index=False)

    print(Fore.YELLOW + f"\n✅ Also saved results to SQLite DB: {sqlite_path}")

    # Step 12: Demonstrate SQL query execution
    print(Fore.CYAN + "\n📊 Clickbait count from SQL query:")

    # Example SQL query to count clickbait vs non-clickbait articles
    query = """
    SELECT Clickbait, COUNT(*) as count
    FROM articles
    GROUP BY Clickbait
    """

    # Execute query and display results
    df_clickbait_counts = pd.read_sql_query(query, conn)
    print(df_clickbait_counts)

    # Close database connection properly
    conn.close()

    # Step 13: Run comprehensive statistical analysis
    run_statistical_analysis(df_results)

# Entry point - run main function when script is executed directly
if __name__ == "__main__":
    main()
