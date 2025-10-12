# backend/server.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from datetime import datetime
from pathlib import Path
import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# dev: allow requests from your frontend. In production, restrict origins.
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_CSV = PROJECT_ROOT.parent / "data/processed/labeled_news_multi.csv"


def _safe_iso_date(value):
    """Return ISO string for dates or None if not parseable."""
    try:
        if pd.isna(value):
            return None
        # If already a datetime
        if isinstance(value, (datetime,)):
            return value.isoformat()
        # Try to parse string-like
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.isoformat()
    except Exception:
        return None


@app.route("/api/news")
def get_news():
    try:
        if not DATA_CSV.exists():
            return jsonify({"error": f"Data file not found: {DATA_CSV}"}), 500

        # read CSV normally (date as string initially)
        df = pd.read_csv(DATA_CSV, dtype=str)  # read as strings to avoid mixed types

        # If there's a 'date' column, coerce it to ISO strings (or None)
        if "date" in df.columns:
            df["date"] = df["date"].apply(_safe_iso_date)

        # Replace any remaining NaN-like values with None so jsonify produces null
        df = df.where(pd.notnull(df), None)

        # Try to sort by date if valid ISO strings present, otherwise keep original order
        try:
            df["__date_parsed"] = df["date"].apply(lambda v: pd.to_datetime(v) if v is not None else pd.NaT)
            df = df.sort_values("__date_parsed", ascending=False).drop(columns="__date_parsed")
        except Exception:
            # fallback: keep file order
            pass

        records = df.to_dict(orient="records")
        # ensure each record has an id
        for i, r in enumerate(records):
            r.setdefault("id", i)

        return jsonify(records)
    except Exception as e:
        logger.exception("Failed to load news")
        return jsonify({"error": "Failed to load news", "details": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        payload = request.get_json(force=True)
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "no text provided"}), 400

        if not hasattr(app, "classifier"):
            logger.info("Loading classifier for /api/analyze")
            app.classifier = pipeline.load_classifier()

        labels, scores = pipeline.predict_sentiment(
            app.classifier, pd.Series([text]), batch_size=1
        )
        return jsonify({
            "text": text,
            "sentiment": labels[0],
            "confidence": float(scores[0]),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.exception("Analyze failed")
        return jsonify({"error": "Analyze failed", "details": str(e)}), 500


if __name__ == "__main__":
    # run pipeline once at startup (produces labeled_news.csv)
    try:
        logger.info("Running initial pipeline.run_pipeline()")
        pipeline.run_pipeline()
    except Exception:
        logger.exception("pipeline.run_pipeline() failed at startup; starting server anyway")

    # schedule pipeline
    scheduler = BackgroundScheduler()
    scheduler.add_job(pipeline.run_pipeline, "interval", minutes=15)
    scheduler.start()

    app.run(host="0.0.0.0", port=5050)
