"""
Large Language Model Helper Module

This module provides AI-powered analysis functions using Hugging Face transformers:
1. Sentiment analysis using DistilBERT model
2. Clickbait detection using rule-based pattern matching

The module automatically detects GPU availability and uses it if available
for faster inference.

Dependencies:
    - transformers: Hugging Face transformer models
    - torch: PyTorch for model inference
"""

from transformers import pipeline
import torch

# Initialize sentiment analysis model
print("⏳ Loading sentiment analysis model...")

# Auto-detect GPU availability for faster inference
# device=0 uses GPU, device=-1 uses CPU
device = 0 if torch.cuda.is_available() else -1

# Load pre-trained DistilBERT model for sentiment analysis
# This model is fine-tuned on Stanford Sentiment Treebank (SST-2)
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

print("✅ Model loaded successfully!")

# Curated list of clickbait trigger words and phrases
# Based on common patterns in sensationalized headlines
CLICKBAIT_WORDS = [
    "shocking", "unbelievable", "you won't believe", "secret", "exposed", "surprising",
    "this will change", "the truth about", "miracle", "guaranteed", "instantly", "click here"
]

def analyze_article_with_local_llm(text):
    """
    Analyze article sentiment using a pre-trained transformer model.
    
    This function uses DistilBERT (a smaller, faster version of BERT) to classify
    the sentiment of article text as either Positive or Negative with a confidence score.
    
    Args:
        text (str): Article text to analyze (will be truncated to 512 chars)
        
    Returns:
        dict: Dictionary containing:
              - label (str): "Positive" or "Negative"
              - score (float): Confidence score between 0.0 and 1.0
              
    Example:
        >>> result = analyze_article_with_local_llm("This is great news!")
        >>> print(result)
        {'label': 'Positive', 'score': 0.95}
    """
    # Truncate text to avoid model token limits (512 chars ≈ 100-150 tokens)
    text = text[:512]

    # Run sentiment analysis using the pre-loaded model
    result = classifier(text)[0]

    # Extract and format results
    label = result["label"].capitalize()  # "POSITIVE" → "Positive"
    score = round(float(result["score"]), 2)  # Round to 2 decimal places

    # Return structured result compatible with both main.py and streamlit_app.py
    return {"label": label, "score": score}

def detect_clickbait_in_title(title):
    """
    Detect potential clickbait characteristics in article titles.
    
    This function uses a rule-based approach to identify common clickbait patterns
    by checking for sensationalized words and phrases commonly used in misleading
    or exaggerated headlines.
    
    Args:
        title (str): Article title to analyze
        
    Returns:
        str: "Clickbait" if clickbait patterns detected, "Not Clickbait" otherwise
        
    Example:
        >>> detect_clickbait_in_title("You won't believe this shocking discovery!")
        "Clickbait"
        >>> detect_clickbait_in_title("Scientists publish new research findings")
        "Not Clickbait"
    """
    # Convert to lowercase for case-insensitive matching
    title_lower = title.lower()

    # Check if any clickbait words/phrases are present in the title
    found_words = [word for word in CLICKBAIT_WORDS if word in title_lower]

    # Classify based on presence of clickbait patterns
    if found_words:
        result = "Clickbait"
    else:
        result = "Not Clickbait"

    return result
