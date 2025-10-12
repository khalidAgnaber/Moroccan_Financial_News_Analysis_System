import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from scraper import collect_perfect_financial_news
import preprocess_data
import label_correct  # your relabel‐only script

PROJECT_ROOT     = Path(__file__).parent.parent.resolve()
SCRAPED_CSV      = PROJECT_ROOT / "data/raw/moroccan_financial_news.csv"
DEFAULT_PROC_CSV = PROJECT_ROOT / "data/processed/processed_financial_news.csv"
LABELED_CSV      = PROJECT_ROOT / "data/processed/labeled_news_multi.csv"

HF_MODEL_ID = "bardsai/finance-sentiment-fr-base"
_CLASSIFIER = None

def load_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        from transformers import pipeline
        _CLASSIFIER = pipeline(
            "text-classification",
            model=HF_MODEL_ID,
            tokenizer=HF_MODEL_ID,
            framework="pt",
            return_all_scores=False,
        )
    return _CLASSIFIER

def predict_sentiment(classifier, texts: pd.Series, batch_size: int = 16):
    all_labels = []
    all_scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size].tolist()
        results = classifier(batch, truncation=True)
        for res in results:
            lbl = res["label"].lower()
            if lbl.startswith("pos"):
                all_labels.append("positive")
            elif lbl.startswith("neg"):
                all_labels.append("negative")
            else:
                all_labels.append("neutral")
            all_scores.append(float(res["score"]))
    return all_labels, all_scores

def run_pipeline():
    os.chdir(str(PROJECT_ROOT))

    # 1) Collecting/Scraping
    print("Collecting financial news")
    collect_perfect_financial_news()
    if not SCRAPED_CSV.exists():
        raise FileNotFoundError(f"Expected scraped file at {SCRAPED_CSV}")

    # 2) Preprocessing
    print("Preprocessing data")
    preprocess_data.main()
    if not DEFAULT_PROC_CSV.exists():
        raise FileNotFoundError(f"Expected processed CSV at {DEFAULT_PROC_CSV}")

    # 3) Labeling
    print("Labeling data")
    label_correct.main()
    if not LABELED_CSV.exists():
        raise FileNotFoundError(f"Expected labeled CSV at {LABELED_CSV}")

    # 4) Scraping with timestamp
    df = pd.read_csv(LABELED_CSV)
    df["scraped_at"] = datetime.utcnow().isoformat()
    df.to_csv(LABELED_CSV, index=False)

    print(f"Pipeline completed — final file at: {LABELED_CSV} ({len(df)} rows)")

if __name__ == "__main__":
    run_pipeline()
