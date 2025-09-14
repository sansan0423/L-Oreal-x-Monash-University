import pandas as pd
from relevance import keyword_relevant, map_category, load_zero_shot_model, zero_shot_relevant

# --- Config ---
# --- Config ---
DATA_PATH = "../comments_scored.csv"   # change if needed
TEXT_COL = "textOriginal"             # adjust if your column is named differently
USE_ZERO_SHOT = False              # set True to test zero-shot relevance

# --- Load data ---
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", len(df), "rows")

# --- Apply keyword-based relevance & categories ---
df["is_relevant_keyword"] = df[TEXT_COL].apply(keyword_relevant)
df["category"] = df[TEXT_COL].apply(map_category)

# --- Zero-Shot relevance ---
if USE_ZERO_SHOT:
    print("Loading zero-shot model (this may take a minute)...")
    z_model = load_zero_shot_model()
    df["is_relevant_zeroshot"] = df[TEXT_COL].apply(lambda x: zero_shot_relevant(z_model, x))

# --- Preview results ---
print("\n--- Sample Results ---")
cols = [TEXT_COL, "is_relevant_keyword", "category"]
if USE_ZERO_SHOT:
    cols.append("is_relevant_zeroshot")
print(df[cols].head(10))

# --- Save to CSV for inspection ---
df.to_csv("relevance_test_results.csv", index=False)
print("\n Results saved to relevance_test_results.csv")
