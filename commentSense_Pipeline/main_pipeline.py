import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Imports from your modules ---
from superficial import is_superficial
from relevance import keyword_relevant, map_category
from category import classify_keywords
from measure_kpi import compute_kpis
from visualize_kpis import visualize
from sentiment_analysis import vader_sentiment
# --- Config ---
INPUT_FILE  = "comments1.csv"
OUTPUT_COMMENTS = "comments_scored.csv"
OUTPUT_KPIS     = "overall_kpis.csv"
NUM_PROCESSES = max(1, cpu_count() - 1)  # leave 1 CPU free

# --- Multiprocessing helper ---
def process_chunk(chunk):
    return [vader_sentiment(text) for text in chunk]

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

    # 4) Sentiment Analysis (multiprocessing)
    print(f"Running VADER sentiment predictions on {NUM_PROCESSES} processes...")
    texts = df["textOriginal"].tolist()
    chunk_size = len(texts) // NUM_PROCESSES + 1
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

    with Pool(NUM_PROCESSES) as pool:
        results_chunks = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Sentiment scoring"))

    # Flatten results
    results = [item for sublist in results_chunks for item in sublist]
    labels, scores = zip(*results)
    df["sentiment_label_3class"] = labels
    df["sentiment_score"] = scores

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

if __name__ == "__main__":
    main()
