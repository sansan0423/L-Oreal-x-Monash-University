import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/sansan/L-Oreal-x-Monash-University/comments_scored.csv")
    return df

df = load_data()

st.set_page_config(page_title="💬 CommentSense Dashboard", layout="wide")
st.title("💬 CommentSense Dashboard")
st.markdown("Interactive dashboard for comment quality, sentiment, and engagement insights.")

# --- KPIs ---
st.header("Key Metrics")
qcr = ((df["Relevance"]=="Relevant") & (df["Substantive"]=="Substantive")).mean()
superficial_ratio = (df["Substantive"]=="Superficial").mean()

if "likeCount" in df.columns:
    weighted_qcr = (df.loc[(df["Relevance"]=="Relevant") & (df["Substantive"]=="Substantive"), "likeCount"].sum() + 1) / (df["likeCount"].sum() + 1)
else:
    weighted_qcr = qcr

col1, col2, col3 = st.columns(3)
col1.metric("QCR", f"{qcr:.1%}")
col2.metric("Weighted QCR", f"{weighted_qcr:.1%}")
col3.metric("Superficial %", f"{superficial_ratio:.1%}")

st.markdown("---")

# --- Sentiment Pie ---
st.header("Sentiment Distribution")
sent_counts = df["sentiment_label_3class"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(sent_counts, labels=sent_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
ax1.set_title("Sentiment Breakdown")
st.pyplot(fig1)

st.markdown("---")

# --- Category Bar ---
st.header("Category Distribution")
cat_counts = df["Category"].value_counts()
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.barplot(x=cat_counts.index, y=cat_counts.values, palette="Blues_d", ax=ax2)
ax2.set_ylabel("Count")
ax2.set_xlabel("Category")
ax2.set_title("Category Distribution")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig2)

st.markdown("---")

# --- Likes by Sentiment (if available) ---
if "likeCount" in df.columns:
    st.header("Likes by Sentiment")
    likes_sent = df.groupby("sentiment_label_3class")["likeCount"].sum()
    fig3, ax3 = plt.subplots(figsize=(6,4))
    sns.barplot(x=likes_sent.index, y=likes_sent.values, palette="Greens_d", ax=ax3)
    ax3.set_ylabel("Total Likes")
    ax3.set_xlabel("Sentiment")
    ax3.set_title("Likes by Sentiment")
    st.pyplot(fig3)

st.markdown("---")

# --- Top Comments ---
st.header("Top Comments by Likes")
if "likeCount" in df.columns:
    top_comments = df.sort_values("likeCount", ascending=False).head(5)
    st.table(top_comments[["textOriginal","likeCount","sentiment_label_3class","Category"]])
else:
    st.write("No likeCount column available.")

st.markdown("---")

# --- Raw Data Viewer ---
st.header("Raw Data Viewer")
st.dataframe(df.head(50))
