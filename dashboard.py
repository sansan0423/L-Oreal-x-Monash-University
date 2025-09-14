import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("comments_scored.csv")
    return df

df = load_data()

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

# --- Sentiment Pie ---
st.header("Sentiment Distribution")
sent_counts = df["sentiment_label_3class"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(sent_counts, labels=sent_counts.index, autopct="%1.1f%%", startangle=90)
st.pyplot(fig1)

# --- Category Bar ---
st.header("Category Distribution")
cat_counts = df["Category"].value_counts()
fig2, ax2 = plt.subplots()
cat_counts.plot(kind="bar", color="skyblue", ax=ax2, edgecolor="black")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# --- Likes by Sentiment (if available) ---
if "likeCount" in df.columns:
    st.header("Likes by Sentiment")
    likes_sent = df.groupby("sentiment_label_3class")["likeCount"].sum()
    fig3, ax3 = plt.subplots()
    likes_sent.plot(kind="bar", color="lightgreen", ax=ax3, edgecolor="black")
    ax3.set_ylabel("Total Likes")
    st.pyplot(fig3)

# --- Top Comments ---
st.header("Top Comments by Likes")
if "likeCount" in df.columns:
    top_comments = df.sort_values("likeCount", ascending=False).head(5)
    st.table(top_comments[["textOriginal","likeCount","sentiment_label_3class","Category"]])
else:
    st.write("No likeCount column available.")

# --- Raw Data Viewer ---
st.header("Raw Data")
st.dataframe(df.head(50))
