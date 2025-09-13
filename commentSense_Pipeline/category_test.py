import pandas as pd
from category import classify_keywords, load_zero_shot_model, classify_zero_shot

# --- Sample test comments ---
comments = [
    "I love this serum, my skin feels hydrated!",                   # skincare
    "The lipstick shade is gorgeous but too expensive",              # makeup + price/value
    "This shampoo reduced my frizz",                                 # hair
    "The perfume bottle design is beautiful",                        # fragrance + packaging
    "I like that this product is eco friendly and refillable",       # sustainability
    "This scrub makes my body feel smooth",                          # body
    "L’Oréal always has the best beauty campaigns",                  # brand (if you want to add)
    "Bro when’s the next football match?"                            # irrelevant -> other
]

# --- Create DataFrame ---
df = pd.DataFrame({"comment": comments})

# --- Keyword classification ---
df["Keyword_Categories"] = df["comment"].apply(classify_keywords)

# --- Zero-shot classification ---
print("Loading zero-shot model (this may take ~1 min the first time)...")
z_model = load_zero_shot_model()
df["ZeroShot_Categories"] = df["comment"].apply(lambda x: classify_zero_shot(z_model, x, top_n=2))

# --- Show results ---
print("\n--- Classification Results ---")
print(df)

# --- Save for inspection ---
df.to_csv("category_test_results.csv", index=False)
print("\n✅ Results saved to category_test_results.csv")
