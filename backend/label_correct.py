import pandas as pd
from pathlib import Path
import torch
from transformers import pipeline

HF_MODEL_ID = "bardsai/finance-sentiment-fr-base"

def load_classifier():
    return pipeline(
        "text-classification",
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        framework="pt",
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=False,
        truncation=True
    )

def normalize_label(hf_label: str) -> str:
    l = hf_label.lower()
    if l.startswith("pos"):
        return "positive"
    if l.startswith("neg"):
        return "negative"
    return "neutral"

def main():
    project_root = Path(__file__).parent.resolve()
    input_csv  = project_root / "../data/processed/processed_financial_news.csv"
    output_csv = project_root / "../data/processed/labeled_news.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_csv}")

    print(f"Loading preprocessed data from {input_csv}")
    df = pd.read_csv(input_csv)

    # if output already exists, read and merge
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        # identify new rows by url+title+text
        key = df.apply(lambda row: (row.get('url',''), row['title'], row['text']), axis=1)
        existing_keys = existing.apply(lambda row: (row.get('url',''), row['title'], row['text']), axis=1).tolist()
        mask_new = [k not in existing_keys for k in key]
        if not any(mask_new):
            print("No new articles to label. Exiting.")
            return
        to_label = df[mask_new].copy().reset_index(drop=True)
        base = existing
        print(f"Found {len(base)} existing labeled rows, {len(to_label)} new to label.")
    else:
        to_label = df.copy().reset_index(drop=True)
        base = None
        print(f"No existing labeled file, labeling all {len(df)} rows.")

    classifier = load_classifier()

    batch_size = 16
    combined_texts = (to_label["title"].fillna("") + ". " + to_label["text"].fillna("")).tolist()

    print(f"Re-labeling {len(combined_texts)} articles in batches of {batch_size}")
    new_labels = []
    new_scores = []

    for i in range(0, len(combined_texts), batch_size):
        batch = combined_texts[i : i + batch_size]
        preds = classifier(batch)
        for p in preds:
            new_labels.append(normalize_label(p["label"]))
            new_scores.append(float(p["score"]))

    assert len(new_labels) == len(to_label), "label count mismatch"

    to_label["sentiment"] = new_labels
    to_label["sentiment_score"] = new_scores

    # combine back
    if base is not None:
        final = pd.concat([base, to_label], ignore_index=True)
    else:
        final = to_label

    print("Writing out updated labeled CSV")
    final.to_csv(output_csv, index=False)
    print(f"Done! {len(final)} total rows in {output_csv}")


if __name__ == "__main__":
    main()