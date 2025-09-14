import pandas as pd
from tqdm import tqdm

# --- Imports from your modules ---
from superficial import is_superficial
from relevance import keyword_relevant, map_category
from category import classify_keywords
from transformers import pipeline
from measure_kpi import compute_kpis
from visualize_kpis import visualize

# --- Config ---
INPUT_FILE  = "comments1.csv"
OUTPUT_COMMENTS = "comments_scored.csv"
OUTPUT_KPIS     = "overall_kpis.csv"

MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
DEVICE   = -1   # CPU=-1, GPU=0
BATCH_SIZE = 32

# --- Helper ---
def collapse_label(label: str) -> str:
    """Collapse 5-class labels into 3-class (pos/neu/neg)."""
    label = label.lower()
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    return "neutral"

def main():
    # 1) Load dataset
    df = pd.read_csv(INPUT_FILE)
    if "textOriginal" not in df.columns:
        raise ValueError("Expected a 'textOriginal' column in the CSV.")
    df["textOriginal"] = df["textOriginal"].astype(str)

    print(f"Loaded {len(df):,} comments from {INPUT_FILE}.")

    # 2) Relevance & Category
    print("Classifying relevance & categories...")
    df["Relevance"] = df["textOriginal"].apply(lambda x: "Relevant" if keyword_relevant(x) else "Not Relevant")
    df["Category"] = df["textOriginal"].apply(map_category)

    # 3) Substantive vs Superficial
    print("Checking for superficial comments...")
    df["Substantive"] = df["textOriginal"].apply(lambda x: "Superficial" if is_superficial(x) else "Substantive")

    # 4) Sentiment Analysis
    print("Loading sentiment model...")
    clf = pipeline("sentiment-analysis", model=MODEL_ID, device=DEVICE)

    print("Running sentiment predictions...")
    texts = df["textOriginal"].tolist()
    labels, scores = [], []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Sentiment scoring"):
        batch = texts[i:i+BATCH_SIZE]
        results = clf(batch, truncation=True)
        labels.extend([r["label"].lower() for r in results])
        scores.extend([r["score"] for r in results])

    df["sentiment_label_5class"] = labels
    df["sentiment_score"] = scores
    df["sentiment_label_3class"] = df["sentiment_label_5class"].apply(collapse_label)

    # 5) Save enriched comments
    df.to_csv(OUTPUT_COMMENTS, index=False)
    print(f"Saved per-comment results → {OUTPUT_COMMENTS}")

    # 6) Compute KPIs
    print("Computing KPIs...")
    kpis = compute_kpis(df)
    kpis.to_csv(OUTPUT_KPIS, index=False)
    print(f"Saved overall KPIs → {OUTPUT_KPIS}")

    # 7) Visualize KPIs
    print("Generating visuals...")
    visualize(df)

    # Preview
    preview_cols = ["textOriginal","Relevance","Substantive","Category","sentiment_label_3class","sentiment_score"]
    print("\nSample enriched comments:")
    print(df[preview_cols].head(10).to_string(index=False))

    # Save partial results every N batches
    df_partial = df.iloc[:len(labels)].copy()
    df_partial["sentiment_label_5class"] = labels
    df_partial["sentiment_score"] = scores
    df_partial["sentiment_label_3class"] = df_partial["sentiment_label_5class"].apply(collapse_label)

    df_partial.to_csv("partial_results.csv", index=False)

if __name__ == "__main__":
    main()
