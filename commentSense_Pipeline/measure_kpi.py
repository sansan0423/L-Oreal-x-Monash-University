def compute_kpis(df):
    kpis = {}

    # Fill missing likes with 0
    if "likeCount" in df.columns:
        df["likeCount"] = df["likeCount"].fillna(0)

    # --- QCR (normal + weighted) ---
    quality_mask = (df["Relevance"]=="Relevant") & (df["Substantive"]=="Substantive")
    kpis["QCR"] = quality_mask.mean()

    if "likeCount" in df.columns:
        weighted_qcr = (df.loc[quality_mask, "likeCount"].sum() + quality_mask.sum()) / (df["likeCount"].sum() + len(df))
        kpis["QCR_weighted"] = weighted_qcr

    # --- Superficial % ---
    kpis["Superficial_Ratio"] = (df["Substantive"]=="Superficial").mean()

    # --- Sentiment Ratios ---
    sentiment_dist = df["sentiment_label_3class"].value_counts(normalize=True)
    for s, val in sentiment_dist.items():
        kpis[f"Sentiment_{s}"] = val

    if "likeCount" in df.columns:
        likes_by_sent = df.groupby("sentiment_label_3class")["likeCount"].sum()
        for s, val in likes_by_sent.items():
            kpis[f"Likes_{s}"] = val

    # --- Category Distribution ---
    cat_dist = df["Category"].value_counts(normalize=True)
    for c, val in cat_dist.items():
        kpis[f"Category_{c}"] = val

    if "likeCount" in df.columns:
        likes_by_cat = df.groupby("Category")["likeCount"].sum()
        for c, val in likes_by_cat.items():
            kpis[f"Likes_{c}"] = val

    return pd.DataFrame([kpis])
