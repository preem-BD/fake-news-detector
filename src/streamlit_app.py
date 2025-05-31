"""
Streamlit Web Application for Fake News Detection

This module provides a user-friendly web interface for the Fake News Detector
using Streamlit. It offers the same functionality as the terminal app but with
interactive web-based UI, real-time visualizations, and better user experience.

Features:
    - Interactive web form for keyword search
    - Real-time article analysis and visualization
    - Clickable article selection
    - Pie chart visualization of results
    - HTML-based keyword highlighting
    - Session state management for persistent data
    - CSV export functionality

Usage:
    streamlit run streamlit_app.py

Authors:
    Group 4 — Kai Bleuel, Mustafa Sivgin, César Diaz Murga
    Scientific Programming WIN (2025-FS) - ZHAW
"""

# Import our custom modules
import streamlit as st
from api_fetch import fetch_articles
from clean_data import articles_to_df
from llm_helper import analyze_article_with_local_llm, detect_clickbait_in_title

# Import standard libraries
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def highlight_keyword_html(text, keyword):
    """
    Highlight keyword occurrences in text using HTML/CSS styling.
    
    This function creates HTML-formatted text with red highlighting for
    keyword matches, suitable for display in Streamlit's web interface.
    
    Args:
        text (str): Text to search and highlight in
        keyword (str): Keyword to highlight (case-insensitive)
        
    Returns:
        str: HTML-formatted text with highlighted keywords
        
    Example:
        >>> highlight_keyword_html("Apple iPhone news", "iphone")
        'Apple <span style="color:red; font-weight:bold;">iPhone</span> news'
    """
    if not text:
        return ""
    
    # Create regex pattern for case-insensitive matching
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    
    # Replace matches with HTML span tags for red highlighting
    highlighted_text = pattern.sub(
        r'<span style="color:red; font-weight:bold;">\g<0></span>', 
        text
    )
    
    return highlighted_text

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

# Configure page settings
st.set_page_config(page_title="📰 Fake News Detector", layout="wide")

# ============================================================================
# USER INTERFACE HEADER
# ============================================================================

st.title("📰 Fake News Detector - Streamlit Edition")
st.markdown("#### Group 4 — Kai Bleuel, Mustafa Sivgin, César Diaz Murga")

st.markdown("Detect **Clickbait** and **Sentiment** in real news articles. Enter a topic to begin:")

# ============================================================================
# USER INPUT FORM
# ============================================================================

# Create form for topic input with search button
with st.form(key="topic_form"):
    topic_input = st.text_input("Enter a topic:", "")
    submit_button = st.form_submit_button("🔍 Search")

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

# Initialize session state variables to persist data across reruns
if "selected_article_idx" not in st.session_state:
    st.session_state.selected_article_idx = None
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "df_filtered" not in st.session_state:
    st.session_state.df_filtered = None
if "topic" not in st.session_state:
    st.session_state.topic = ""

# ============================================================================
# ARTICLE SEARCH AND ANALYSIS
# ============================================================================

if submit_button and topic_input.strip() != "":
    topic = topic_input.strip().lower()
    st.session_state.topic = topic
    st.session_state.selected_article_idx = None  # reset selection

    st.info(f"Searching for articles about **'{topic}'**...")

    articles = fetch_articles(topic, page_size=20)

    if not articles:
        st.error(f"No articles found for topic: '{topic}'.")
    else:
        df = articles_to_df(articles)

        df_filtered = df[
            df["title"].str.contains(topic, case=False, na=False) |
            df["text"].str.contains(topic, case=False, na=False)
        ]

        if df_filtered.empty:
            st.error(f"No articles found that actually contain '{topic}'.")
        else:
            st.success(f"✅ Found {len(df_filtered)} article(s) about '{topic}'.")

            results = []

            for idx, (i, row) in enumerate(df_filtered.iterrows(), start=1):
                clickbait_result = detect_clickbait_in_title(row.get('title', ''))
                clickbait_label = "Clickbait" if "likely clickbait" in clickbait_result.lower() else "Not Clickbait"

                sentiment_result = analyze_article_with_local_llm(row["text"])
                sentiment_label = sentiment_result["label"].capitalize()
                sentiment_score = round(sentiment_result["score"], 2)

                combined_label = f"{clickbait_label} + {sentiment_label}"

                word_count = len(row.get('text', '').split())

                results.append({
                    "Article": f"{idx}",
                    "Title": row.get('title', ''),
                    "Clickbait": clickbait_label,
                    "Sentiment": sentiment_label,
                    "Sentiment Score": sentiment_score,
                    "Combined Label": combined_label,
                    "Word Count": word_count
                })

            df_results = pd.DataFrame(results)
            st.session_state.df_results = df_results
            st.session_state.df_filtered = df_filtered

# ---- IF RESULTS EXIST ----

if st.session_state.df_results is not None and st.session_state.df_filtered is not None:
    df_results = st.session_state.df_results
    df_filtered = st.session_state.df_filtered
    topic = st.session_state.topic

    # --- PIE CHART ---
    label_counts = df_results["Combined Label"].value_counts()

    with st.expander("📊 Clickbait + Sentiment Distribution (Click to expand/collapse)", expanded=True):
        fig, ax = plt.subplots(figsize=(3, 3))  # smaller size!
        ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
        ax.set_title(f"Clickbait + Sentiment distribution for '{topic}'")
        st.pyplot(fig, use_container_width=False)

    # --- ARTICLE SELECTION ---
    st.markdown("## 📜 Select an article to analyze:")

    for idx, row in df_results.iterrows():
        button_label = f"[{row['Article']}] {row['Title']}"
        if st.button(button_label):
            st.session_state.selected_article_idx = idx

# ---- IF ARTICLE SELECTED ----

if st.session_state.selected_article_idx is not None:
    idx = st.session_state.selected_article_idx
    selected_row = st.session_state.df_results.iloc[idx]
    df_full = st.session_state.df_filtered.iloc[idx]

    # --- DISPLAY SELECTED ARTICLE ---
    st.markdown("---")
    st.markdown(f"### 🔍 Analyzing Article [{selected_row['Article']}]:")
    title_highlighted = highlight_keyword_html(selected_row["Title"], st.session_state.topic)
    st.markdown(f"**📰 Title:** {title_highlighted}", unsafe_allow_html=True)

    # --- FULL TEXT ---
    full_text = df_full.get('text') or df_full.get('content') or df_full.get('description') or df_full.get('title', '')
    text_highlighted = highlight_keyword_html(full_text, st.session_state.topic)
    st.markdown("**📝 Full Text:**", unsafe_allow_html=True)
    st.markdown(text_highlighted, unsafe_allow_html=True)

    # --- FAKE CHECK RESULTS ---

    # Clickbait result color
    clickbait_color = "green" if selected_row['Clickbait'] == "Not Clickbait" else "red"
    st.markdown(f"🚩 **CLICKBAIT RESULT:** <span style='color:{clickbait_color}; font-weight:bold;'>{selected_row['Clickbait'].upper()}</span>", unsafe_allow_html=True)

    # Sentiment result color
    if selected_row['Sentiment'] == "Positive":
        sentiment_color = "green"
    elif selected_row['Sentiment'] == "Negative":
        sentiment_color = "red"
    else:
        sentiment_color = "gray"
    st.markdown(f"🎭 **SENTIMENT RESULT:** <span style='color:{sentiment_color}; font-weight:bold;'>{selected_row['Sentiment'].upper()}</span> (score: {selected_row['Sentiment Score']})", unsafe_allow_html=True)

    # Word count
    st.markdown(f"📝 **WORD COUNT:** `{selected_row['Word Count']}`")

    # --- SAVE CSV ---
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "combined_analysis_streamlit.csv")
    st.session_state.df_results.to_csv(output_path, index=False)

    st.success(f"✅ Combined results saved to: {output_path}")
    st.info("👉 You can rerun with another topic, or try different articles!")

# ---- FOOTER ----

st.markdown("---")
st.markdown("### ℹ️ **Instructions for Terminal:**")
st.markdown("To run this app in terminal, use: `streamlit run src/streamlit_app.py`")

# ---- COMPANY FOOTER ----

st.markdown("---")
st.markdown("**Fake-Checkers AG**  \nBahnhofstrasse 1, 8001 Zürich  \nPhone: +41 79 123 45 67  \nEmail: fake-checker@check.ch")
st.markdown("**© 2023 Fake-Checkers AG**  \nAll rights reserved. This app is for educational purposes only.")