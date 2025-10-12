#!/usr/bin/env python3
"""
label_branches_multi.py

Assigns one or more branches per article (semicolon-separated).
- Uses expanded rule-based keyword scoring (title weighted).
- If rules give no signal, uses zero-shot multi_label classification (joeddav/xlm-roberta-large-xnli).
- Writes:
  - data/processed/labeled_news_multi.csv (final)
  - data/processed/labeled_news_multi_review.csv (rows flagged for manual review)
"""

import re
from pathlib import Path
import logging
import argparse
import pandas as pd
import time
from collections import defaultdict
import torch
from transformers import pipeline as hf_pipeline

# ---------- Paths ----------
ROOT = Path(__file__).parent.parent.resolve()
INPUT_CSV = ROOT / "../data/processed/labeled_news.csv"
OUTPUT_CSV = ROOT / "../data/processed/labeled_news_multi.csv"
REVIEW_CSV = ROOT / "../data/processed/labeled_news_multi_review.csv"

# ---------- Zero-shot config ----------
ZS_MODEL = "joeddav/xlm-roberta-large-xnli"
# Candidate labels (in French). Map them to canonical branch keys.
ZS_LABELS_FR = ["finance d'entreprise", "gestion d'actifs", "private equity", "courtage et conservation", "immobilier"]
ZS_MAP = {
    "finance d'entreprise": "corporate finance",
    "gestion d'actifs": "asset management",
    "private equity": "private equity",
    "courtage et conservation": "brokerage&custody",
    "immobilier": "real estate"
}

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("label_branches_multi")

# ---------- Large keyword lists (expanded) ----------
BRANCH_KEYWORDS = {
    "corporate finance": [
        "fusion", "acquisition", "rachat", "cession", "prise de participation", "opa", "opv",
        "offre publique", "introduction en bourse", "ipo", "augmentation de capital", "émission d'actions",
        "levée de fonds", "obligation", "émission obligataire", "prêt syndiqué", "refinancement",
        "restructuration financière", "résultat", "résultats", "chiffre d'affaires", "bénéfice net",
        "dividende", "rachat d'actions", "notation", "banque", "banque d'affaires", "financement d'entreprise"
    ],
    "asset management": [
        "gestion d'actifs", "gestion dactifs", "gestion de portefeuille", "asset management",
        "fonds", "sicav", "fcp", "opcvm", "etf", "allocation d'actifs",
        "performance des fonds", "rendement des fonds", "gestion institutionnelle", "assurance vie",
        "gestion collective", "gestion privée", "gestion de patrimoine"
    ],
    "private equity": [
        "capital investissement", "capital-investissement", "private equity", "venture capital",
        "levée de fonds", "seed", "amorçage", "series a", "buy-out", "lbo", "mezzanine",
        "limited partner", "general partner", "carried interest", "préamorçage", "pré-amorçage"
    ],
    "brokerage&custody": [
        "courtage", "brokerage", "courtier", "teneur de marché", "market making", "exécution d'ordres",
        "plateforme de trading", "clearing", "règlement-livraison", "conservation de titres",
        "maroclear", "custody", "dépositaire", "compte titre", "repo", "margin", "prime brokerage"
    ],
    "real estate": [
        "immobilier", "foncier", "promotion immobilière", "investissement immobilier", "foncière",
        "reit", "sci", "scpi", "opci", "pierre-papier", "crédit immobilier", "hypothèque", "loyer",
        "rendement locatif", "prix immobilier", "transaction immobilière", "retail", "bureaux", "entrepôt",
        "hôtellerie", "logistique", "cap rate", "surface m²", "m²"
    ]
}

# compile regex patterns; sort longer phrases first
COMPILED = {}
for b, kw_list in BRANCH_KEYWORDS.items():
    sorted_kw = sorted(set(kw_list), key=lambda s: -len(s))
    COMPILED[b] = [re.compile(r"\b" + re.escape(k.lower()) + r"\b", flags=re.IGNORECASE) for k in sorted_kw]

# heuristics
MONEY_RE = re.compile(r"\b\d{1,3}(?:[ \.,]\d{3})*(?:[ \.,]\d+)?\s*(?:milliards?|millions?|mdh|mmdh|dh|dirhams?|euros?|€|usd|dollars?)\b", flags=re.IGNORECASE)
PERCENT_RE = re.compile(r"\b\d+(\.\d+)?\s*(%|pour ?cent)\b", flags=re.IGNORECASE)

# small org keywords list to bias corporate finance
INSTITUTION_KEYWORDS = ["attijariwafa", "bcp", "cih", "bmce", "bmci", "cdg", "maroclear", "banque", "bank"]

# ---------- Helpers ----------
def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.replace("\n", " ")).strip()

def score_row(title, text):
    """
    Compute weighted score for each branch.
    Title matches weight 3, text matches weight 1.
    Return scores dict and hit_counts dict.
    """
    t = normalize_text(title).lower()
    x = normalize_text(text).lower()
    scores = defaultdict(float)
    hits = defaultdict(int)
    for branch, patterns in COMPILED.items():
        for pat in patterns:
            tcount = len(pat.findall(t))
            xcount = len(pat.findall(x))
            if tcount:
                scores[branch] += 3.0 * tcount
                hits[branch] += tcount
            if xcount:
                scores[branch] += 1.0 * xcount
                hits[branch] += xcount
    # institution keywords -> bump corporate finance
    for inst in INSTITUTION_KEYWORDS:
        if inst in t:
            scores["corporate finance"] += 2.0
            hits["corporate finance"] += 1
        if inst in x:
            scores["corporate finance"] += 1.0
            hits["corporate finance"] += 1
    # money/percent signals
    if MONEY_RE.search(t) or MONEY_RE.search(x):
        scores["corporate finance"] += 0.8
        scores["real estate"] += 0.6
    if PERCENT_RE.search(t) or PERCENT_RE.search(x):
        scores["asset management"] += 0.5
    return dict(scores), dict(hits)

def select_multiple_branches(scores, hits):
    """
    Given scores & hits, return list of branch keys (could be 0+).
    Strategy:
      - If no scores >0: return empty list (caller may use zero-shot).
      - Let top_score = max(scores).
      - Include any branch where:
          score >= max(0.8, top_score * 0.5) OR hits >= 2
      - This allows multiple branches when they are near top or have multiple hits.
    """
    if not scores or max(scores.values()) <= 0:
        return []
    top_score = max(scores.values())
    threshold = max(0.8, top_score * 0.5)
    chosen = []
    for branch, sc in scores.items():
        if sc >= threshold or hits.get(branch, 0) >= 2:
            chosen.append(branch)
    # fallback: ensure at least top branch is there
    if not chosen:
        # pick branch(es) with top score
        top = [k for k, v in scores.items() if v == top_score]
        chosen = top
    return sorted(set(chosen))

def load_zero_shot():
    device = 0 if torch.cuda.is_available() else -1
    logger.info("Loading zero-shot model %s on %s", ZS_MODEL, "cuda" if device == 0 else "cpu")
    return hf_pipeline("zero-shot-classification", model=ZS_MODEL, device=device)

def zero_shot_multi(zs, text, multi_threshold=0.35):
    """
    Return a list of mapped branch keys from zero-shot multi-label classification,
    keeping labels with score >= multi_threshold.
    """
    try:
        out = zs(text, ZS_LABELS_FR, multi_label=True)
    except Exception as e:
        logger.exception("Zero-shot failed: %s", e)
        return []
    # out may be dict if single input; normalize
    if isinstance(out, dict):
        out = [out]
    pred = out[0]
    chosen = []
    for lbl, sc in zip(pred["labels"], pred["scores"]):
        if sc >= multi_threshold:
            mapped = ZS_MAP.get(lbl)
            if mapped:
                chosen.append(mapped)
    return sorted(set(chosen)), dict(zip(pred["labels"], pred["scores"]))

# ---------- Main ----------
def main(force=True, zs_threshold=0.35, verbose=False):
    if not INPUT_CSV.exists():
        logger.error("Input CSV not found: %s", INPUT_CSV)
        raise SystemExit(1)

    df = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d rows from %s", len(df), INPUT_CSV)

    df["title"] = df.get("title", "").fillna("").astype(str)
    df["text"] = df.get("text", df.get("processed_text", "")).fillna("").astype(str)
    if "branch" not in df.columns:
        df["branch"] = ""

    zs = None
    ambiguous_rows = []
    assigned = 0
    kept = 0
    start = time.time()

    for idx, row in df.iterrows():
        existing = str(row.get("branch") or "").strip()
        if existing and not force:
            kept += 1
            continue

        title = row["title"]
        text = row["text"]
        scores, hits = score_row(title, text)
        chosen = select_multiple_branches(scores, hits)

        source_note = "rule"
        rule_scores_snapshot = scores.copy()

        if not chosen:
            # use zero-shot multi-label fallback
            if zs is None:
                zs = load_zero_shot()
            input_text = (title or "") + " " + (text or "")
            picked_zs, zs_scores = zero_shot_multi(zs, input_text[:2000], multi_threshold=zs_threshold)
            if picked_zs:
                chosen = picked_zs
                source_note = f"zero_shot"
            else:
                # ultimate fallback: pick top scoring rule branch (even if zero)
                if scores:
                    best = max(scores.items(), key=lambda kv: kv[1])[0]
                    chosen = [best]
                    source_note = "fallback_best_rule"
                else:
                    chosen = ["corporate finance"]
                    source_note = "ultimate_fallback"

        # build string
        branch_str = "; ".join(chosen)
        df.at[idx, "branch"] = branch_str
        df.at[idx, "branch_source"] = source_note
        # flag ambiguous if more than 1 branch or low scores
        if len(chosen) > 1:
            ambiguous_rows.append({
                "index": idx,
                "title": title,
                "url": row.get("url", ""),
                "source": row.get("source", ""),
                "assigned_branches": branch_str,
                "branch_source": source_note,
                "rule_scores": rule_scores_snapshot,
                "hits": hits
            })
        else:
            # if single branch but low confidence, mark for review
            b = chosen[0]
            sc = scores.get(b, 0.0)
            if sc < 1.0 and source_note == "rule":
                ambiguous_rows.append({
                    "index": idx,
                    "title": title,
                    "url": row.get("url", ""),
                    "source": row.get("source", ""),
                    "assigned_branches": branch_str,
                    "branch_source": source_note,
                    "rule_scores": rule_scores_snapshot,
                    "hits": hits
                })

        assigned += 1

    elapsed = time.time() - start
    logger.info("Processed rows: assigned=%d kept=%d time=%.1fs", assigned, kept, elapsed)

    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Wrote output: %s", OUTPUT_CSV)

    if ambiguous_rows:
        review_df = pd.DataFrame(ambiguous_rows)
        review_df.to_csv(REVIEW_CSV, index=False)
        logger.info("Wrote review file: %s (%d rows)", REVIEW_CSV, len(review_df))
    else:
        pd.DataFrame([]).to_csv(REVIEW_CSV, index=False)
        logger.info("No ambiguous rows; wrote empty review file: %s", REVIEW_CSV)

    # log distribution
    dist = df["branch"].value_counts().to_dict()
    logger.info("Final branch distribution (top keys): %s", {k: dist[k] for k in list(dist)[:10]})
    return OUTPUT_CSV, REVIEW_CSV

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="label_branches_multi.py")
    ap.add_argument("--no-force", action="store_true", help="Do not overwrite non-empty branch values")
    ap.add_argument("--zs-threshold", type=float, default=0.35, help="zero-shot label inclusion threshold (0-1)")
    args = ap.parse_args()
    main(force=not args.no_force, zs_threshold=args.zs_threshold)
