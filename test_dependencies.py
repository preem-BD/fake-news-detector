#!/usr/bin/env python3
"""
Simple dependency test script for the Fake News Detector project.
Tests if all required packages can be imported successfully.
"""

def test_imports():
    """Test importing all required packages."""
    try:
        # Core data processing
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import scipy
        print("✅ scipy imported successfully")
        
        # Web requests
        import requests
        print("✅ requests imported successfully")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
        
        # Machine Learning
        import transformers
        print("✅ transformers imported successfully")
        
        import torch
        print("✅ torch imported successfully")
        
        # Visualization
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
        
        import seaborn as sns
        print("✅ seaborn imported successfully")
        
        # Web app framework
        import streamlit as st
        print("✅ streamlit imported successfully")
        
        # Other utilities
        import colorama
        print("✅ colorama imported successfully")
        
        # AI/LLM integration
        import openai
        print("✅ openai imported successfully")
        
        print("\n🎉 ALL DEPENDENCIES IMPORTED SUCCESSFULLY!")
        print("Your Fake News Detector project is ready to run!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
