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
   {
        "name": "Boursenews",
        "url": "https://boursenews.ma/articles/marches",
        "language": "fr"
    },
    {
        "name": "FLM",
        "url": "https://flm.ma/economie.php",
        "language": "fr"
    },
    {
        "name": "Le Desk",
        "url": "https://ledesk.ma/encontinu/",
        "language": "fr"
    },
    {
        "name": "H24Info",
        "url": "https://www.h24info.ma/economie/",
        "language": "fr"
    },
    {
        "name": "Le Matin",
        "url": "https://lematin.ma/economie",
        "language": "fr"
    },
    {
        "name": "Medias24",
        "url": "https://medias24.com/categorie/leboursier/",
        "language": "fr"
    },
    {
        "name": "Finance News - Financial",
        "url": "https://fnh.ma/articles/actualite-financiere-maroc",
        "language": "fr"
    },
    {
        "name": "Finance News - Economic",
        "url": "https://fnh.ma/articles/actualite-economique",
        "language": "fr"
    }
]

# List of major Moroccan companies to track
COMPANIES = [
{"name": "Douja Prom Addoha", "ticker": "ADH", "keywords": ["Douja Prom", "Addoha", "Douja Prom Addoha", "ADH"]},
    {"name": "Alliances", "ticker": "ADI", "keywords": ["Alliances", "Alliances Développement Immobilier", "ADI"]},
    {"name": "Afriquia Gaz", "ticker": "GAZ", "keywords": ["Afriquia Gaz", "GAZ"]},
    {"name": "Agma Lahlou-Tazi", "ticker": "AGM", "keywords": ["Agma", "Agma Lahlou-Tazi", "AGM"]},
    {"name": "Aluminium du Maroc", "ticker": "ALM", "keywords": ["Aluminium du Maroc", "ALM"]},
    {"name": "Atlanta", "ticker": "ATW", "keywords": ["Atlanta", "AtlantaSanad", "ATW"]},
    {"name": "Attijariwafa Bank", "ticker": "ATW", "keywords": ["Attijariwafa", "Attijariwafa Bank", "ATW", "Attijari"]},
    {"name": "Auto Hall", "ticker": "AHL", "keywords": ["Auto Hall", "AHL"]},
    {"name": "Balima", "ticker": "BAL", "keywords": ["Balima", "BAL"]},
    {"name": "Banque Centrale Populaire", "ticker": "BCP", "keywords": ["BCP", "Banque Centrale Populaire"]},
    {"name": "BMCE Bank (Bank of Africa)", "ticker": "BOA", "keywords": ["BMCE", "BMCE Bank", "Bank of Africa", "BOA"]},
    {"name": "BMCI", "ticker": "BCI", "keywords": ["BMCI", "Banque Marocaine pour le Commerce et l'Industrie", "BCI"]},
    {"name": "CDM", "ticker": "CDM", "keywords": ["CDM", "Crédit du Maroc"]},
    {"name": "CIH", "ticker": "CIH", "keywords": ["CIH", "CIH Bank", "Crédit Immobilier et Hôtelier"]},
    {"name": "Miniere Touissit", "ticker": "CMT", "keywords": ["Miniere Touissit", "CMT"]},
    {"name": "Colorado", "ticker": "COL", "keywords": ["Colorado", "COL"]},
    {"name": "Cartier Saada", "ticker": "CRS", "keywords": ["Cartier Saada", "CRS"]},
    {"name": "Cosumar", "ticker": "CSR", "keywords": ["Cosumar", "CSR", "Compagnie Sucrerie Marocaine"]},
    {"name": "CTM", "ticker": "CTM", "keywords": ["CTM"]},
    {"name": "Dari Couspate", "ticker": "DRC", "keywords": ["Dari", "Dari Couspate", "DRC"]},
    {"name": "Delta Holding", "ticker": "DHO", "keywords": ["Delta Holding", "DHO"]},
    {"name": "Delattre Levivier Maroc", "ticker": "DLM", "keywords": ["Delattre Levivier Maroc", "DLM"]},
    {"name": "Societe Equipement", "ticker": "SNE", "keywords": ["Societe Equipement", "SNE"]},
    {"name": "Fenie Brossette", "ticker": "FBR", "keywords": ["Fenie Brossette", "FBR"]},
    {"name": "HPS", "ticker": "HPS", "keywords": ["HPS"]},
    {"name": "Itissalat Al-Maghrib (Maroc Telecom)", "ticker": "IAM", "keywords": ["Maroc Telecom", "Itissalat Al-Maghrib", "IAM"]},
    {"name": "IB Maroc Com", "ticker": "IBM", "keywords": ["IB Maroc", "IB Maroc Com", "IBM"]},
    {"name": "Involys", "ticker": "INV", "keywords": ["Involys", "INV"]},
    {"name": "LafargeHolcim Maroc", "ticker": "LHM", "keywords": ["LafargeHolcim", "Lafarge Holcim Maroc", "LHM", "Holcim", "Lafarge"]},
    {"name": "Label Vie", "ticker": "LBV", "keywords": ["Label'Vie", "Label Vie", "Carrefour Maroc", "LBV"]},
    {"name": "Lesieur Cristal", "ticker": "LES", "keywords": ["Lesieur Cristal", "Lesieur", "LES"]},
    {"name": "M2M Group", "ticker": "M2M", "keywords": ["M2M Group", "M2M"]},
    {"name": "Maghrebail", "ticker": "MAB", "keywords": ["Maghrebail", "MAB"]},
    {"name": "Micro Data SA", "ticker": "MIC", "keywords": ["Micro Data", "Micro Data SA", "MIC"]},
    {"name": "Maroc Leasing", "ticker": "MLI", "keywords": ["Maroc Leasing", "MLI"]},
    {"name": "Managem", "ticker": "MNG", "keywords": ["Managem", "MNG"]},
    {"name": "Auto Nejma", "ticker": "NEJ", "keywords": ["Auto Nejma", "NEJ"]},
    {"name": "Les Eaux Minerales Oulmes", "ticker": "OLM", "keywords": ["Oulmes", "Les Eaux Minerales Oulmes", "OLM"]},
    {"name": "Maghreb Oxygene", "ticker": "MGO", "keywords": ["Maghreb Oxygene", "MGO"]},
    {"name": "Med Paper", "ticker": "PAP", "keywords": ["Med Paper", "PAP"]},
    {"name": "Ste Promotion Pharmaceutique du Maghreb", "ticker": "PROM", "keywords": ["Promotion Pharmaceutique du Maghreb", "PROM"]},
    {"name": "Rebab Company", "ticker": "RBC", "keywords": ["Rebab Company", "RBC"]},
    {"name": "Risma", "ticker": "RIS", "keywords": ["Risma", "RIS"]},
    {"name": "Societe des Boissons du Maroc", "ticker": "SBM", "keywords": ["Boissons du Maroc", "Societe des Boissons du Maroc", "SBM"]},
    {"name": "Ciments Du Maroc", "ticker": "CMA", "keywords": ["Ciments Du Maroc", "CMA"]},
    {"name": "Salafin", "ticker": "SLF", "keywords": ["Salafin", "SLF"]},
    {"name": "SMI", "ticker": "SMI", "keywords": ["SMI", "Société Minière d’Imlil"]},
    {"name": "Stokvis Nord Afrique", "ticker": "STO", "keywords": ["Stokvis Nord Afrique", "Stokvis", "STO"]},
    {"name": "Nationale d’Electrolyse et de Petrochimie Ste", "ticker": "NEP", "keywords": ["Nationale d’Electrolyse et de Petrochimie", "NEP"]},
    {"name": "Ste Nationale de Siderurgie", "ticker": "SNS", "keywords": ["Siderurgie", "Ste Nationale de Siderurgie", "SNS"]},
    {"name": "Marocaine Ste de Therapeutique", "ticker": "MST", "keywords": ["Marocaine Ste de Therapeutique", "MST"]},
    {"name": "Realis. Mecaniques", "ticker": "RM", "keywords": ["Realis. Mecaniques", "RM"]},
    {"name": "Unimer", "ticker": "UNI", "keywords": ["Unimer", "UNI"]},
    {"name": "Wafa Assurance", "ticker": "WAF", "keywords": ["Wafa Assurance", "Wafa", "WAF"]},
    {"name": "Zellidja S.A", "ticker": "ZDJ", "keywords": ["Zellidja", "Zellidja S.A", "ZDJ"]},
    {"name": "Afric Industries Sa", "ticker": "AFI", "keywords": ["Afric Industries", "Afric Industries Sa", "AFI"]},
    {"name": "Sanlam Maroc", "ticker": "SAN", "keywords": ["Sanlam Maroc", "Sanlam", "SAN"]},
    {"name": "Disway", "ticker": "DIS", "keywords": ["Disway", "DIS"]},
    {"name": "Jet Contractors", "ticker": "JET", "keywords": ["Jet Contractors", "JET"]},
    {"name": "Ennakl Automobiles SA", "ticker": "NKL", "keywords": ["Ennakl", "Ennakl Automobiles SA", "NKL"]},
    {"name": "Ste de Travaux de Realisation d’Ouvrages et de Con", "ticker": "TROC", "keywords": ["Travaux Generaux De Construction", "Ste de Travaux de Realisation d’Ouvrages", "TROC"]},
    {"name": "Taqa Morocco SA", "ticker": "TQM", "keywords": ["Taqa Morocco", "Taqa", "Taqa Morocco SA", "TQM"]},
    {"name": "S2M", "ticker": "S2M", "keywords": ["S2M"]},
    {"name": "Residences Dar Saada", "ticker": "RDS", "keywords": ["Residences Dar Saada", "RDS"]},
    {"name": "Total Maroc SA", "ticker": "TMA", "keywords": ["Total Maroc", "Total Maroc SA", "Total", "TMA"]},
    {"name": "AFMA SA", "ticker": "AFM", "keywords": ["AFMA", "AFMA SA", "AFM"]},
    {"name": "Societe d’Exploitation des Ports", "ticker": "MRS", "keywords": ["Societe d’Exploitation des Ports", "Marsa Maroc", "MRS"]},
    {"name": "Mutandis", "ticker": "MUT", "keywords": ["Mutandis", "MUT"]},
    {"name": "Immorente Invest", "ticker": "IMO", "keywords": ["Immorente Invest", "IMO"]},
    {"name": "Aradei Capital", "ticker": "ARAD", "keywords": ["Aradei Capital", "ARAD"]},
    {"name": "Travaux Generaux De Construction", "ticker": "TGCC", "keywords": ["TGCC", "Travaux Generaux De Construction"]},
    {"name": "Disty Tech", "ticker": "DISTY", "keywords": ["Disty Tech", "Disty", "DISTY"]},
    {"name": "Akdital", "ticker": "AKT", "keywords": ["Akdital", "AKT"]},
    {"name": "CFG Bank", "ticker": "CFG", "keywords": ["CFG Bank", "CFG"]},
    {"name": "Stroc Industrie", "ticker": "STR", "keywords": ["Stroc", "Stroc Industrie", "STR"]},
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