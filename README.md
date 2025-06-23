# Moroccan News Sentiment Analysis for Investment Insights

This project analyzes sentiment in Moroccan financial news (in French) to generate investment insights for Moroccan companies. It collects news articles, preprocesses them, analyzes sentiment, and provides investment recommendations (buy, sell, or hold).

## Project Structure

```
moroccan_news_sentiment/
├── README.md
├── requirements.txt
├── config.py                      # Configuration parameters
├── data/                          # Directory to store raw and processed data
│   ├── raw/
│   └── processed/
├── data_collection/
│   ├── __init__.py
│   ├── scraper.py                 # Web scraping logic
│   └── data_storage.py            # Logic to store collected data
├── data_preprocessing/
│   ├── __init__.py
│   ├── cleaner.py                 # Text cleaning functions
│   ├── preprocessor.py            # Text preprocessing functions
│   └── french_stopwords.py        # French stopwords list
├── models/
│   ├── __init__.py
│   ├── base_model.py              # Base model class
│   ├── lstm_model.py              # LSTM implementation
│   ├── bert_model.py              # BERT implementation
│   └── transformer_model.py       # Transformer implementation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # Custom metrics implementation
│   └── visualization.py           # Result visualization
├── investment_recommendation/
│   ├── __init__.py
│   └── recommendation_engine.py   # Logic to generate buy/sell/hold recommendations
├── utils/
│   ├── __init__.py
│   ├── logger.py                  # Logging utilities
│   └── helpers.py                 # Helper functions
└── main.py                        # Main entry point for the application
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khalidAgnaber/moroccan-news-sentiment.git
cd moroccan-news-sentiment
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Running the Full Pipeline

To run the complete pipeline (data collection, preprocessing, training, evaluation, and recommendations):

```bash
python main.py --action pipeline
```

### Individual Steps

1. Collect news data:
```bash
python main.py --action collect
```

2. Preprocess collected data:
```bash
python main.py --action preprocess
```

3. Generate investment recommendations:
```bash
python main.py --action recommend
```

4. Schedule daily data collection:
```bash
python main.py --action schedule
```

## Data Collection

The system collects French financial news from several Moroccan news sources:
- L'Economiste
- Médias24
- Challenge
- Les Eco
- Finance News
- Boursenews

It tracks news about major Moroccan companies listed in the configuration file.

## Data Preprocessing

The preprocessing pipeline includes:
- Text normalization (lowercase, accent removal)
- Stop word removal (using French financial stopwords)
- Removal of URLs, HTML tags, mentions, and hashtags
- Punctuation removal and extra space removal
- Tokenization and padding

## Sentiment Analysis Models

Multiple model architectures are implemented:
1. LSTM model (default)
2. Simple LSTM model (lighter version)
3. Transformer model
4. BERT-based model (using CamemBERT for French language)

## Investment Recommendations

The recommendation engine considers:
- Overall sentiment score of recent news
- Sentiment trends over time
- News recency (more recent news has higher weight)

It generates one of three recommendations:
- BUY: Strong positive sentiment
- HOLD: Neutral or mixed sentiment
- SELL: Strong negative sentiment

Each recommendation includes a confidence score and explanation.

## Evaluation Metrics

Custom evaluation metrics are implemented using TP, TN, FP, FN for each sentiment class, providing:
- Accuracy
- Precision
- Recall
- F1-score

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.