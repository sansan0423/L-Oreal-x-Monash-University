import pandas as pd
from transformers import pipeline

# ----------------- Config -----------------
MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
DEVICE   = -1   # CPU = -1, GPU = 0
# ------------------------------------------

def collapse_label(label: str) -> str:
    """Collapse 5-class labels into 3-class (pos/neu/neg)."""
    label = label.lower()
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    return "neutral"

def main():
    # --- Sample comments (English + Malay + Mixed) ---
    comments = [
        "Highly recommend this product, it really works!",          # very positive
        "I was disappointed, it did not work as advertised.",       # very negative
        "Produk ni okay la, but I expected better quality.",        # neutral / mixed
        "Saya sangat suka bau perfume ni, memang tahan lama!",      # positive (Malay)
        "Harga dia terlalu mahal, not worth it.",                   # negative + price/value
        "🔥🔥🔥",                                                    # superficial, likely neutral
    ]

    df = pd.DataFrame({"comment": comments})

    # Load model
    print("Loading sentiment model...")
    clf = pipeline("sentiment-analysis", model=MODEL_ID, device=DEVICE)

    # Run predictions
    results = clf(df["comment"].tolist(), truncation=True)
    df["sentiment_label_5class"] = [r["label"].lower() for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    df["sentiment_label_3class"] = df["sentiment_label_5class"].apply(collapse_label)

    # Show results
    print("\n--- Sentiment Test Results ---")
    print(df)

    # Save for inspection
    df.to_csv("sentiment_test_results.csv", index=False)
    print("\n✅ Results saved → sentiment_test_results.csv")

if __name__ == "__main__":
    main()

