"""
Configuration settings for the Moroccan News Sentiment Analysis project.
"""
import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models" / "saved"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data collection settings
NEWS_SOURCES = [
    {"name": "L'Economiste", "url": "https://www.leconomiste.com/", "language": "fr"},
    {"name": "Médias24", "url": "https://www.medias24.com/", "language": "fr"},
    {"name": "Les Eco", "url": "https://leseco.ma/", "language": "fr"},
    {"name": "Finance News", "url": "https://fnh.ma/", "language": "fr"},
    {"name": "Boursenews", "url": "https://www.boursenews.ma/", "language": "fr"},
]

# List of major Moroccan companies to track
COMPANIES = [
    {"name": "Attijariwafa Bank", "ticker": "ATW", "keywords": ["Attijariwafa", "Attijariwafa Bank"]},
    {"name": "Maroc Telecom", "ticker": "IAM", "keywords": ["Maroc Telecom", "IAM", "Itissalat Al-Maghrib"]},
    {"name": "Bank of Africa", "ticker": "BOA", "keywords": ["Bank of Africa", "BMCE", "BOA"]},
    {"name": "Cosumar", "ticker": "CSR", "keywords": ["Cosumar", "CSR"]},
    {"name": "Lafarge Holcim Maroc", "ticker": "LHM", "keywords": ["Lafarge", "Holcim", "LafargeHolcim"]},
    {"name": "Label Vie", "ticker": "LBV", "keywords": ["Label'Vie", "Label Vie", "Carrefour Maroc"]},
    {"name": "Marsa Maroc", "ticker": "MRS", "keywords": ["Marsa Maroc", "SODEP"]},
    {"name": "Taqa Morocco", "ticker": "TQM", "keywords": ["Taqa Morocco", "Taqa"]},
    {"name": "Sonasid", "ticker": "SID", "keywords": ["Sonasid"]},
    {"name": "Managem", "ticker": "MNG", "keywords": ["Managem"]},
]

# Data collection frequency
COLLECTION_INTERVAL_HOURS = 24

# Data preprocessing settings
MAX_SEQUENCE_LENGTH = 512
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# Model settings
LSTM_EMBEDDING_DIM = 200
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# BERT model settings
BERT_MODEL_NAME = "camembert-base"  # CamemBERT for French language
BERT_MAX_LENGTH = 256

# Evaluation metrics
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

# Investment recommendation settings
# Number of days to consider for trend analysis
TREND_WINDOW = 14
# Weights for recency of news (more recent news has higher impact)
RECENCY_WEIGHT_DECAY = 0.9
# Thresholds for sentiment scores to generate recommendations
POSITIVE_THRESHOLD = 0.6
NEGATIVE_THRESHOLD = 0.4