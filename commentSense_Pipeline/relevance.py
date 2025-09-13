from transformers import pipeline

# --- Keyword lists 
SKINCARE_KW = ["skin", "skincare", "serum", "moisturizer", "hydration", "sunscreen", "spf", "cream"]
MAKEUP_KW = ["makeup", "foundation", "concealer", "mascara", "lipstick", "eyeshadow", "blush", "eyeliner"]
FRAGRANCE_KW = ["perfume", "fragrance", "scent", "smell", "cologne"]
HAIRCARE_KW = ["hair", "shampoo", "conditioner", "dye", "color", "colour", "styling", "frizz", "treatment"]
BRAND_KW = ["loreal", "l'oréal", "lorealparis"]

# --- Basic keyword relevance check ---
def keyword_relevant(raw_text: str) -> bool:
    """
    Returns True if raw_text contains any beauty-related keywords.
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    raw = raw_text.lower()
    keywords = SKINCARE_KW + MAKEUP_KW + FRAGRANCE_KW + HAIRCARE_KW + BRAND_KW
    return any(kw in raw for kw in keywords)

# --- Category mapping ---
def map_category(raw_text: str) -> str:
    """
    Assigns a category to the comment based on keywords in raw_text.
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    raw = raw_text.lower()

    if any(kw in raw for kw in SKINCARE_KW):
        return "Skincare"
    if any(kw in raw for kw in MAKEUP_KW):
        return "Makeup"
    if any(kw in raw for kw in FRAGRANCE_KW):
        return "Fragrance"
    if any(kw in raw for kw in HAIRCARE_KW):
        return "Haircare"
    if any(kw in raw for kw in BRAND_KW):
        return "Brand"
    return "Other"

# --- zero-shot classifier for relevance ---
def load_zero_shot_model():
    """Load zero-shot classification model."""
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def zero_shot_relevant(model, raw_text: str) -> bool:
    """
    Classify relevance using zero-shot model.
    Returns True if 'relevant' is predicted.
    """
    try:
        result = model(raw_text, candidate_labels=["relevant", "not relevant"])
        return result["labels"][0] == "relevant"
    except:
        return False
