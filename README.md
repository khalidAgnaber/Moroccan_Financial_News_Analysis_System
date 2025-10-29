# Moroccan Financial News Analysis System

A comprehensive full-stack application for analyzing sentiment in Moroccan financial news articles. The system collects, processes, and analyzes French-language financial news from various Moroccan sources, providing real-time sentiment analysis and investment insights.

## 🌟 Features

- **Automated News Collection**: Web scraping from multiple Moroccan financial news sources
- **Advanced Text Processing**: French language preprocessing with financial domain optimization
- **Sentiment Analysis**: BERT-based sentiment classification (positive, neutral, negative)
- **Branch Classification**: Multi-label classification for financial sectors (banking, markets, corporate finance, etc.)
- **Interactive Dashboard**: Real-time news feed with sentiment visualization
- **Manual Review Tool**: Interactive labeling interface for data quality assurance
- **RESTful API**: Flask backend with sentiment analysis endpoints

## 🏗️ Project Structure

Moroccan_Financial_News_Analysis_System/
├── backend/
│   ├── scraper.py                 # News collection from multiple sources
│   ├── text_cleaner.py            # Financial text cleaning utilities
│   ├── text_preprocessor.py       # French NLP preprocessing
│   ├── embedding_utils.py         # FastText word embeddings
│   ├── french_stopwords.py        # Custom French stopwords for finance
│   ├── pipeline.py                # End-to-end processing pipeline
│   ├── server.py                  # Flask API server
│   ├── label_correct.py           # Sentiment labeling logic
│   └── tools/
│       └── label_branches_auto.py # Automated branch classification
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LandingPage.tsx    # Auth0 login page
│   │   │   ├── Dashboard.tsx      # Main news dashboard
│   │   │   ├── LandingPage.css
│   │   │   └── Dashboard.css
│   │   ├── App.tsx                # Root component
│   │   └── index.tsx              # React entry point
│   └── public/
├── data/
│   ├── raw/                       # Scraped news data
│   └── processed/                 # Preprocessed and labeled data
├── label/
│   ├── labeling_tool.py           # Interactive review tool
│   └── auto_sentiment_labeler.py  # Automated labeling
├── models/
│   ├── bert_model.py              # BERT sentiment classifier
│   └── train_bert.py              # Model training script
├── evaluation/
│   └── metrics.py                 # Custom evaluation metrics
├── utils/
│   ├── logger.py                  # Logging utilities
│   └── helpers.py                 # Helper functions
├── config.py                      # Configuration settings
├── main.py                        # CLI entry point
└── requirements.txt

## 📋 Prerequisites
Python 3.8+
Node.js 16+
Auth0 account (for authentication)

## 🚀 Installation
Backend Setup
1. Clone the repository:
git clone https://github.com/khalidAgnaber/Moroccan_Financial_News_Analysis_System.git
cd Moroccan_Financial_News_Analysis_System

2. Create and activate virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Python dependencies:
pip install -r requirements.txt

4. Install spaCy French model:
python -m spacy download fr_core_news_md

5. Download NLTK data:
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

## Frontend Setup
1. Navigate to frontend directory:
cd frontend

2. Install dependencies:
npm install

3. Create .env file with Auth0 credentials:
REACT_APP_AUTH0_DOMAIN=your-auth0-domain
REACT_APP_AUTH0_CLIENT_ID=your-auth0-client-id

## 🎯 Usage
Running the Complete Pipeline
1. Start the backend server:
cd backend
python server.py

This will:

Run the data collection pipeline
Process and label the news data
Start the Flask API on http://localhost:5001

2. Start the frontend:
cd frontend
npm start

The application will open at http://localhost:3000

Individual Components
# Data Collection
python backend/scraper.py
Collects news from configured sources and saves to moroccan_financial_news.csv

# Data Preprocessing
python backend/preprocess_data.py
Processes raw text using FinancialTextCleaner and FinancialNewsPreprocessor

# Manual Labeling Tool
python label/labeling_tool.py data/processed/labeled_news.csv
Interactive terminal interface for reviewing and correcting sentiment labels

# Automated Branch Classification
python backend/tools/label_branches_auto.py
Assigns financial sector branches using rule-based + zero-shot classification

## 📊 Data Sources
The system collects news from:

Finance News
Medias24
Boursenews
Le Desk
H24Info
FLM

## 🧠 Models & Technologies
# Backend
NLP: spaCy, NLTK, FastText embeddings
ML: Transformers (CamemBERT), scikit-learn
Framework: Flask, APScheduler
Data: pandas, BeautifulSoup4, newspaper3k
# Frontend
Framework: React 18 + TypeScript
Authentication: Auth0
Styling: Custom CSS with animations
HTTP Client: Fetch API

## 🔍 Key Features Explained
Sentiment Analysis
Uses bardsai/finance-sentiment-fr-base BERT model fine-tuned for French financial sentiment with three classes:

Positive: Growth, profits, positive market movements
Neutral: Factual reporting without clear direction
Negative: Losses, declines, risks

Branch Classification
Multi-label classification across 5 sectors

Text Preprocessing
Specialized pipeline for French financial text:

Preserves monetary values (€, $, DH, MDH)
Preserves percentages and financial ratios
Handles French apostrophes and contractions
Custom stopwords excluding financial terms
Entity recognition for companies and institutions
Interactive Dashboard
Real-time news feed with sentiment badges
Branch classification tags
Expandable article view
Manual sentiment analysis for custom text
Responsive design with financial theme

## 📈 Data Processing Pipeline
1. Scraping → 2. Cleaning → 3. Preprocessing → 4. Labeling → 5. Storage
     ↓             ↓              ↓               ↓             ↓
  Raw HTML    Normalized    Tokenized      Sentiment      CSV/API
              Text          Embeddings     + Branch

## Detailed Steps:

1. Scraping (scraper.py)
Fetches articles from configured sources
Filters financial content using keyword matching
Removes duplicates via content hashing
Classifies into preliminary branches

2. Cleaning (text_cleaner.py)
Normalizes Unicode characters
Removes HTML entities and extra whitespace
Preserves financial entities (€, $, DH, MDH, percentages)

3. Preprocessing (text_preprocessor.py)
Tokenization with spaCy French model
Lemmatization and POS tagging
Custom stopword removal (preserves financial terms)
Named entity recognition

4. Labeling (label_correct.py)
BERT-based sentiment classification
Multi-label branch assignment
Manual review for edge cases

5. Storage (data_storage.py)
CSV export with timestamps
Backup management
Date range filtering

## 🧠 Models & Technologies
Backend

NLP: spaCy (fr_core_news_md), NLTK, FastText embeddings
ML: Transformers (CamemBERT, XLM-RoBERTa), scikit-learn
Framework: Flask, Flask-CORS, APScheduler
Data: pandas, BeautifulSoup4, newspaper3k, requests
ML Frameworks: TensorFlow (macOS Metal optimized)
Frontend

Framework: React 18 + TypeScript
Authentication: Auth0 React SDK
Styling: Custom CSS with animations and gradients
HTTP Client: Fetch API
Build: Create React App

## 🔍 Key Features Explained
Sentiment Analysis
Uses bardsai/finance-sentiment-fr-base BERT model fine-tuned for French financial sentiment with three classes:

🟢 Positive: Growth, profits, positive market movements, economic expansion
🔵 Neutral: Factual reporting without clear directional bias
🔴 Negative: Losses, declines, risks, market downturns
The model achieves ~88% accuracy on the test set with:

Precision: 71-92% across classes
High confidence predictions: 87.6% of total
Class-weighted training for imbalanced data
Branch Classification
Multi-label classification across 15+ financial sectors defined in label_branches_auto.py:

Text Preprocessing
Specialized pipeline for French financial text via FinancialNewsPreprocessor:

Preserves Financial Entities:

Monetary values (€, $, DH, MDH, MMDH)
Percentages and financial ratios
Stock tickers and company names
French Language Handling:

Handles apostrophes and contractions (l', d', qu')
Custom stopwords excluding financial terms
Lemmatization with spaCy
Entity Recognition:

Company detection from COMPANIES (75+ Moroccan companies)
Financial institution identification
Market index recognition (MASI, MADEX)

Interactive Dashboard
Built with React + TypeScript (Dashboard.tsx):

Real-time News Feed:

Sentiment badges with color coding
Branch classification tags
Expandable article view
Source links and timestamps
Manual Sentiment Analysis:

Paste custom text for instant sentiment analysis
Confidence scores
Analysis history
Design Features:

Animated financial background with stock tickers
Responsive grid layout
Glassmorphism UI effects
Mobile-optimized

Automated Scraping
FinancialNewsScraper with intelligent filtering:

Content Validation:

Minimum word count (100 words)
Financial keyword density checks
Non-financial term exclusion
URL pattern validation
Duplicate Detection:

Content hash comparison
URL deduplication
Rate Limiting:

Configurable delays between requests
Retry logic with exponential backoff
Performance Tracking:

Articles processed/accepted/rejected stats
Branch distribution metrics
Processing time monitoring

## 🛠️ Configuration
Edit config.py to customize:

COMPANIES = [
    {"name": "Attijariwafa Bank", "ticker": "ATW", "keywords": [...]},
    {"name": "Maroc Telecom", "ticker": "IAM", "keywords": [...]},
    {"name": "LafargeHolcim Maroc", "ticker": "LHM", "keywords": [...]},
    # ... 72 more companies
]

Data Directories
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "saved"

News Sources
NEWS_SOURCES = [
    {"name": "Boursenews", "url": "https://boursenews.ma/...", "language": "fr"},
    {"name": "Medias24", "url": "https://medias24.com/...", "language": "fr"},
    # ... more sources
]

Model Parameters (in train_bert.py)
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 3e-5
MAX_LENGTH = 384
PATIENCE = 4

## 🔒 Authentication
The frontend uses Auth0 for secure authentication (LandingPage.tsx):

Users land on the landing page with company branding
Click "Se connecter" to authenticate via Auth0
Redirected to Dashboard after successful login
Logout button available in dashboard header

# Auth0 Setup:
Create an Auth0 application (Single Page Application)
Add http://localhost:3000 to Allowed Callback URLs
Add http://localhost:3000 to Allowed Logout URLs
Copy Domain and Client ID to .env

## 📊 Evaluation Metrics
Custom metrics implementation in metrics.py:

Accuracy: Overall correct predictions
Precision: Correct positive predictions / All positive predictions
Recall: Correct positive predictions / All actual positives
F1-Score: Harmonic mean of precision and recall
Confusion Matrix: Detailed prediction breakdown
Class-wise Performance: Per-class metrics for positive/neutral/negative

Example output from train_bert.py:

precision    recall  f1-score   support
    negative     71.21%     66.67%     68.88%        15
     neutral     87.69%     86.76%     87.22%        68
    positive     91.87%     92.74%     92.30%       135

    accuracy                         88.59%       218

## 🎨 UI/UX Features
# Landing Page (LandingPage.tsx)
Animated geometric background shapes
Company logo and branding
Feature highlights with icons
Glassmorphism login card
Responsive design

# Dashboard (Dashboard.tsx)
Stock Ticker Animation: Scrolling financial symbols
Sentiment Color Coding: 🟢 Green (Positive), 🔵 Blue (Neutral), 🔴 Red (Negative)
Branch Badges: Multi-label sector tags with custom styling
Expandable Cards: "Lire la suite" / "Afficher moins" toggle
Real-time Analysis: Instant sentiment feedback on custom text
Dark Theme: Financial dashboard aesthetic with gradient accents
Mobile Responsive: Grid layout adapts to screen size

Styling Highlights (Dashboard.css, LandingPage.css)

/* Glassmorphism effect */
backdrop-filter: blur(10px);
background: rgba(45, 55, 72, 0.9);

/* Gradient text shimmer */
animation: textShimmer 4s ease-in-out infinite alternate;

/* Card hover effects */
transform: translateY(-2px);
box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);

## 🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

## 📧 Contact
For questions, issues, or collaboration:

Open an issue on GitHub
Email: khalid.agnaber.ka@gmail.com
LinkedIn: Khalid Agnaber


Note: This is an educational/research project for analyzing Moroccan financial news sentiment. Not intended as financial advice. Always consult with qualified financial professionals before making investment decisions.