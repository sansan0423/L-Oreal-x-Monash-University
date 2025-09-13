import pandas as pd
import matplotlib.pyplot as plt

def visualize(df):
    # --- QCR Pie ---
    labels = ["Relevant & Substantive", "Other"]
    qcr_val = ((df["Relevance"]=="Relevant") & (df["Substantive"]=="Substantive")).mean()
    plt.figure(figsize=(5,5))
    plt.pie([qcr_val, 1-qcr_val], labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Quality Comment Ratio (QCR)")
    plt.savefig("kpi_qcr.png")
    plt.close()

    # --- Sentiment Pie (3-class) ---
    sentiment_counts = df["sentiment_label_3class"].value_counts(normalize=True)
    plt.figure(figsize=(5,5))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Sentiment Distribution")
    plt.savefig("kpi_sentiment.png")
    plt.close()

    # --- Substantive vs Superficial Pie ---
    subs_counts = df["Substantive"].value_counts(normalize=True)
    plt.figure(figsize=(5,5))
    plt.pie(subs_counts, labels=subs_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Substantive vs Superficial")
    plt.savefig("kpi_superficial.png")
    plt.close()

    # --- Category Bar Chart ---
    cat_counts = df["Category"].value_counts(normalize=True).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    cat_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Category Distribution")
    plt.ylabel("Proportion of Comments")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("kpi_categories.png")
    plt.close()

    print("✅ Visual KPIs saved (png files).")

if __name__ == "__main__":
    df = pd.read_csv("comments_scored.csv")
    visualize(df)
