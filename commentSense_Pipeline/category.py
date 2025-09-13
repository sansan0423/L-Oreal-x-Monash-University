from transformers import pipeline


# --- Keyword lists ---
CATEGORIES = {
    "skincare": ["skin", "skincare", "serum"],
    "makeup": ["makeup", "foundation", "concealer"],
    "fragrance": ["perfume", "fragrance", "scent"],
    "hair": ["hair", "shampoo", "conditioner"],
    "body": ["body lotion", "scrub", "butter", "cream", "oil"],
    "packaging": ["bottle", "pump", "cap", "tube", "jar", "design", "packaging"],
    "sustainability": ["eco", "green", "recycle", "sustainable", "environment", "refill"],
    "price/value": ["price", "expensive", "cheap", "worth it", "value", "affordable"]
}

# --- Keyword classifier ---
def classify_keywords(raw_text: str):
    """
    Multi-label classification using keywords.
    Returns a list of categories that matched.
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    raw = raw_text.lower()

    matched = []
    for cat, kws in CATEGORIES.items():
        if any(kw in raw for kw in kws):
            matched.append(cat)

    return matched if matched else ["other"]

# --- Zero-shot classifier ---
def load_zero_shot_model():
    """Load Hugging Face zero-shot classification model."""
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_zero_shot(model, raw_text: str, top_n=1):
    """
    Classify using zero-shot model.
    Returns top-N categories.
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    try:
        labels = list(CATEGORIES.keys())
        result = model(raw_text, candidate_labels=labels, multi_label=True)
        # Sort by score (highest first)
        sorted_labels = sorted(
            zip(result["labels"], result["scores"]),
            key=lambda x: x[1],
            reverse=True
        )
        return [lbl for lbl, score in sorted_labels[:top_n]]
    except Exception as e:
        print("Zero-shot error:", e)
        return ["other"]