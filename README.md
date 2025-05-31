# Fake News Detector & Article Analyzer

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![AI](https://img.shields.io/badge/AI-Transformers-orange.svg)
![Database](https://img.shields.io/badge/database-SQLite-lightblue.svg)

**Group 4 — Kai Bleuel, Mustafa Sivgin, César Diaz Murga**  
Scientific Programming WIN (2025-FS) - ZHAW

## 🎯 Project Overview

An automated tool that analyzes real-world news articles for:
- **Clickbait detection** using rule-based pattern matching
- **Sentiment analysis** using Hugging Face Transformer models
- **Statistical analysis** with correlation tests and p-values
- **Interactive web interface** built with Streamlit

## 🚀 Features

- ✅ **Real-time data collection** via NewsAPI
- ✅ **AI-powered sentiment analysis** using DistilBERT
- ✅ **Clickbait detection** with curated keyword patterns
- ✅ **Statistical analysis** (Pearson, Spearman, Chi-Square tests)
- ✅ **Data visualization** with interactive charts
- ✅ **SQLite database** integration with SQL queries
- ✅ **Web application** for interactive analysis
- ✅ **Keyword highlighting** in articles

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- `pandas` - Data manipulation and analysis
- `streamlit` - Web application framework
- `transformers` - Hugging Face models for sentiment analysis
- `torch` - PyTorch for transformer models
- `requests` - API calls to NewsAPI
- `matplotlib` - Data visualization
- `scipy` - Statistical analysis
- `colorama` - Terminal text coloring
- `python-dotenv` - Environment variable management

## 🚀 Quick Start

**Want to try it immediately? Use our demo script (no API keys needed):**

```bash
# 1. Clone and install
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt

# 2. Run demo with sample data
python src/demo.py
```

**For full functionality with real-time data:**

```bash
# 1. Get free NewsAPI key from https://newsapi.org/
# 2. Create .env file with your API key
echo "NEWS_API_KEY=your_key_here" > .env

# 3. Run the applications
python src/main.py              # Terminal interface
streamlit run src/streamlit_app.py  # Web interface
```

## 🔧 Detailed Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```
   NEWS_API_KEY=your_newsapi_key_here
   OPENAI_API_KEY=your_openai_key_here  # Optional
   ```
   
   Get your free NewsAPI key from: https://newsapi.org/

4. **Run the application:**

   **Terminal version:**
   ```bash
   python src/main.py
   ```
   
   **Web application:**
   ```bash
   streamlit run src/streamlit_app.py
   ```

## 📊 Usage

### Terminal Application
1. Run `python src/main.py`
2. Enter a keyword to search for news articles
3. View analysis results with colored output
4. Check statistical correlations and p-values

### Web Application
1. Run `streamlit run src/streamlit_app.py`
2. Enter keyword in the sidebar
3. Explore interactive visualizations
4. Download analysis results as CSV

## 🗃️ Database Integration

The application automatically:
- Saves analysis results to SQLite database (`data/fake_news_analysis.db`)
- Executes SQL queries for data aggregation
- Stores articles with clickbait detection and sentiment scores

## 📈 Statistical Analysis

The tool performs three statistical tests:
1. **Pearson Correlation** - Linear relationship between sentiment and word count
2. **Spearman Correlation** - Monotonic relationship between variables  
3. **Chi-Square Test** - Independence between clickbait and sentiment categories

## 🏗️ Project Structure

```
fake_news_detecter/
├── src/
│   ├── main.py              # Terminal application
│   ├── streamlit_app.py     # Web application
│   ├── api_fetch.py         # NewsAPI integration
│   ├── llm_helper.py        # Sentiment analysis with LLM
│   ├── clean_data.py        # Data cleaning utilities
│   └── analyze.py           # Analysis functions
├── notebooks/
│   ├── FakeNewsAnalysis.ipynb  # Jupyter analysis notebook
│   └── images/              # Screenshots and visualizations
├── data/                    # Analysis results and database
├── requirements.txt         # Python dependencies
└── .env                     # API keys (not in repo)
```

## 🎓 Academic Requirements Met

This project fulfills all requirements for Scientific Programming WIN:

### Minimum Requirements (8 points):
- ✅ Real-world data collection (NewsAPI)
- ✅ Data preparation with regex and pandas
- ✅ Python data structures (lists, dicts, DataFrames)
- ✅ Control flow (loops, conditionals)
- ✅ Procedural programming
- ✅ Data visualization (pie charts, statistics)
- ✅ Statistical analysis with p-values
- ✅ Code availability

### Bonus Features (6 points):
- ✅ Creative implementation (clickbait + sentiment analysis)
- ✅ Web API integration (NewsAPI)
- ✅ Database with SQL queries (SQLite)
- ✅ Large Language Model usage (Hugging Face)
- ✅ Web application (Streamlit)
- ✅ GitHub repository

## 🔍 Research Question

**Can we identify patterns between clickbait headlines, sentiment and article length in real-world news?**

Our analysis reveals interesting insights about media patterns across different topics and news sources.

## 📸 Screenshots

See the `notebooks/images/` directory for:
- Terminal output examples
- Streamlit web interface
- Statistical analysis results
- Data visualizations

## 🤝 Contributors

- **Kai Bleuel** - Data analysis and statistical implementation
- **Mustafa Sivgin** - Web API integration and database design  
- **César Diaz Murga** - LLM integration and web interface

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Created for academic purposes as part of the Scientific Programming WIN course at ZHAW.

## 📧 Contact

For questions about this project, please contact the contributors through the course platform.
