"""
OpenAI API Test Module

This is a simple test script to verify that the OpenAI API integration
is working correctly. It makes a basic API call to test connectivity
and API key validity.

This file is optional and used for testing purposes only.
The main project uses Hugging Face transformers for sentiment analysis,
not OpenAI's API.

Usage:
    python test_openai.py

Environment Variables Required:
    OPENAI_API_KEY: Your OpenAI API key (stored in .env file)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client using the new v1.0+ syntax
client = OpenAI(api_key=api_key)

try:
    # Make a simple test API call to GPT-3.5-turbo
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Summarize what AI is in one sentence."}
        ],
        temperature=0.5  # Controls randomness (0.0 = deterministic, 1.0 = very random)
    )

    # Print successful response
    print("✅ GPT Response:\n", response.choices[0].message.content)

except Exception as e:
    # Handle any errors (invalid API key, network issues, etc.)
    print("❌ Error:", e)
