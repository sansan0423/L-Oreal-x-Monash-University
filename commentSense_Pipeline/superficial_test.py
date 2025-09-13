import pandas as pd
from superficial import is_superficial

# --- Sample test comments ---
comments = [
    "🔥🔥🔥",             # emoji only
    "nice",                # generic
    "good product",        # too short
    "Love it",             # filler phrase
    "This foundation blends really well and lasts all day",  # substantive
    "Produk ni memang bagus, sangat berbaloi dengan harga"  # substantive (Malay)
]

df = pd.DataFrame({"comment": comments})
df["is_superficial"] = df["comment"].apply(is_superficial)

print(df)
