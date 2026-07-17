"""
Restaurant Review Intelligence — Majestic Café (Porto) case study + live analysis mode.

Run locally with: streamlit run app.py
Deploy on Hugging Face Spaces: push this file + requirements.txt + the 3 data files
(aspect_sentiment_summary.csv, executive_summary.txt, majestic_cafe_reviews_with_sentiment.csv)
to a new Space (SDK: Streamlit).
"""

import streamlit as st
import pandas as pd
import ast
import re
import time
import plotly.express as px

st.set_page_config(page_title="Restaurant Review Intelligence", layout="wide", page_icon="🍽️")

# ──────────────────────────────────────────────────────────────────────────
# ASPECT KEYWORDS (shared by both precomputed data and live pipeline)
# ──────────────────────────────────────────────────────────────────────────
ASPECT_KEYWORDS = {
    "service": ["service", "staff", "waiter", "waitress", "server", "rude", "friendly", "attentive"],
    "food": ["food", "toast", "coffee", "cake", "croissant", "soup", "sandwich", "tasty", "delicious", "flavour", "flavor"],
    "price": ["price", "expensive", "overpriced", "pricey", "cheap", "€", "euro", "cost", "worth"],
    "ambience": ["atmosphere", "ambience", "ambiance", "decor", "architecture", "beautiful", "historic", "vibe", "interior"],
    "wait_crowds": ["queue", "wait", "crowd", "busy", "line", "packed", "rush"],
}

def extract_aspects(text):
    text_lower = text.lower()
    return [a for a, kws in ASPECT_KEYWORDS.items() if any(kw in text_lower for kw in kws)]

def compute_aspect_summary(df):
    exploded = df.explode("aspects_mentioned").rename(columns={"aspects_mentioned": "aspect"})
    exploded = exploded[exploded["aspect"].notna()]
    if exploded.empty:
        return pd.DataFrame()
    return (
        exploded.groupby("aspect")["sentiment_label"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
        .round(2)
    )

# ──────────────────────────────────────────────────────────────────────────
# CACHED MODEL LOADERS (only loaded once per app session)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    from transformers import pipeline, AutoTokenizer
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    # use_fast=False avoids needing the Rust-based "tokenizers" package to build from
    # source (which fails on very new Python versions) — falls back to the pure-Python
    # slow tokenizer, which only needs sentencepiece.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
    )

# ──────────────────────────────────────────────────────────────────────────
# PRECOMPUTED DATA LOADER (Majestic Café case study)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_precomputed():
    df = pd.read_csv("majestic_cafe_reviews_with_sentiment.csv")
    aspect_summary = pd.read_csv("aspect_sentiment_summary.csv", index_col=0)
    with open("executive_summary.txt", "r") as f:
        summary_text = f.read()
    # aspects_mentioned was saved as a stringified list — parse it back
    df["aspects_mentioned"] = df["aspects_mentioned"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
    )
    return df, aspect_summary, summary_text

# ──────────────────────────────────────────────────────────────────────────
# LIVE PIPELINE (any restaurant, run on demand)
# ──────────────────────────────────────────────────────────────────────────
def fetch_reviews_live(restaurant_query, serpapi_key, max_pages_per_sort=3):
    from serpapi import GoogleSearch

    place_params = {"engine": "google_maps", "q": restaurant_query, "api_key": serpapi_key}
    place_result = GoogleSearch(place_params).get_dict()
    place = place_result.get("place_results")
    if not place:
        return None, None
    data_id = place["data_id"]
    place_name = place.get("title", restaurant_query)

    def fetch(sort_by=None, max_pages=max_pages_per_sort):
        all_reviews = []
        params = {"engine": "google_maps_reviews", "data_id": data_id, "api_key": serpapi_key}
        if sort_by:
            params["sort_by"] = sort_by
        token = None
        for _ in range(max_pages):
            if token:
                params["next_page_token"] = token
            data = GoogleSearch(params).get_dict()
            reviews = data.get("reviews", [])
            if not reviews:
                break
            all_reviews.extend(reviews)
            token = data.get("serpapi_pagination", {}).get("next_page_token")
            if not token:
                break
            time.sleep(0.5)
        return all_reviews

    default_reviews = fetch(sort_by=None)
    low_reviews = fetch(sort_by="ratingLow")
    newest_reviews = fetch(sort_by="newestFirst")

    combined = pd.DataFrame(default_reviews + low_reviews + newest_reviews)
    if combined.empty:
        return None, place_name
    combined = combined.drop_duplicates(subset="review_id", keep="first")
    return combined, place_name

def clean_and_prepare(raw_df):
    def get_original_text(row):
        try:
            extracted = row.get("extracted_snippet")
            if pd.notna(extracted):
                parsed = extracted if isinstance(extracted, dict) else ast.literal_eval(extracted)
                if isinstance(parsed, dict) and parsed.get("original"):
                    return parsed["original"]
        except (ValueError, SyntaxError):
            pass
        return row.get("snippet", "") if pd.notna(row.get("snippet", "")) else ""

    df = raw_df.copy()
    df["review_text"] = df.apply(get_original_text, axis=1)
    df = df[df["review_text"].astype(str).str.strip().str.len() > 10].copy()

    def clean_text(text):
        text = re.sub(r"http\S+|www\S+", "", str(text))
        return re.sub(r"\s+", " ", text).strip()

    df["review_text_clean"] = df["review_text"].apply(clean_text)
    return df

def run_live_pipeline(restaurant_query, serpapi_key, groq_key):
    raw_df, place_name = fetch_reviews_live(restaurant_query, serpapi_key)
    if raw_df is None:
        return None

    df = clean_and_prepare(raw_df)
    if df.empty:
        return None

    sentiment_model = load_sentiment_model()
    results = sentiment_model(df["review_text_clean"].tolist(), batch_size=16)
    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    df["aspects_mentioned"] = df["review_text_clean"].apply(extract_aspects)

    aspect_summary = compute_aspect_summary(df)

    # LLM summary via Groq
    from groq import Groq
    client = Groq(api_key=groq_key)

    overall_counts = df["sentiment_label"].value_counts()
    overall_pct = (overall_counts / overall_counts.sum() * 100).round(1)

    pos = df[df["sentiment_label"] == "positive"]["review_text_clean"]
    neg = df[df["sentiment_label"] == "negative"]["review_text_clean"]
    pos_quotes = pos.sample(min(4, len(pos)), random_state=1).tolist() if len(pos) else []
    neg_quotes = neg.sample(min(4, len(neg)), random_state=1).tolist() if len(neg) else []

    def shorten(t, n=200):
        return t[:n] + ("..." if len(t) > n else "")

    prompt = f"""You are a business analyst preparing a short executive summary about
customer feedback for {place_name}, based on {len(df)} Google reviews analyzed.

OVERALL SENTIMENT: {overall_pct.to_dict()}

ASPECT-LEVEL SENTIMENT (proportion of mentions that are negative/neutral/positive):
{aspect_summary.to_string() if not aspect_summary.empty else "Not enough aspect data."}

SAMPLE POSITIVE QUOTES:
{chr(10).join(f"- {shorten(q)}" for q in pos_quotes)}

SAMPLE NEGATIVE QUOTES:
{chr(10).join(f"- {shorten(q)}" for q in neg_quotes)}

Write a concise executive summary (150-200 words) for the restaurant's management, covering:
1. Overall sentiment picture
2. The strongest aspect
3. The weakest aspect / main complaint
4. One concrete, actionable recommendation

Tone: direct, professional, no fluff. Interpret the numbers, don't just restate them.
"""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    summary_text = completion.choices[0].message.content

    return {"df": df, "aspect_summary": aspect_summary, "summary_text": summary_text, "place_name": place_name}

# ──────────────────────────────────────────────────────────────────────────
# SHARED DASHBOARD RENDERER
# ──────────────────────────────────────────────────────────────────────────
def render_dashboard(df, aspect_summary, summary_text, title):
    st.subheader(title)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews analyzed", len(df))
    col2.metric("Avg. rating", f"{df['rating'].mean():.2f} ★" if "rating" in df.columns else "—")
    pos_pct = (df["sentiment_label"] == "positive").mean() * 100
    col3.metric("% Positive", f"{pos_pct:.0f}%")
    neg_pct = (df["sentiment_label"] == "negative").mean() * 100
    col4.metric("% Negative", f"{neg_pct:.0f}%")

    st.info(summary_text)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Overall sentiment**")
        sent_counts = df["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig = px.pie(sent_counts, names="sentiment", values="count", hole=0.45,
                     color="sentiment",
                     color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Sentiment by aspect**")
        if not aspect_summary.empty:
            asp = aspect_summary.reset_index().melt(id_vars="aspect", var_name="sentiment", value_name="proportion")
            fig2 = px.bar(asp, x="aspect", y="proportion", color="sentiment", barmode="stack",
                          color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("Not enough data to break down by aspect.")

    if "iso_date" in df.columns:
        st.markdown("**Sentiment over time**")
        try:
            df_time = df.copy()
            df_time["date"] = pd.to_datetime(df_time["iso_date"], errors="coerce")
            df_time = df_time.dropna(subset=["date"])
            df_time["year"] = df_time["date"].dt.year
            trend = df_time.groupby(["year", "sentiment_label"]).size().reset_index(name="count")
            fig3 = px.line(trend, x="year", y="count", color="sentiment_label", markers=True,
                           color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"})
            st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            pass

    st.markdown("**Browse reviews**")
    fc1, fc2 = st.columns(2)
    sentiment_filter = fc1.multiselect("Sentiment", options=df["sentiment_label"].unique().tolist(),
                                        default=df["sentiment_label"].unique().tolist())
    all_aspects = sorted({a for lst in df["aspects_mentioned"] for a in lst}) if "aspects_mentioned" in df.columns else []
    aspect_filter = fc2.multiselect("Aspect mentioned", options=all_aspects)

    filtered = df[df["sentiment_label"].isin(sentiment_filter)]
    if aspect_filter:
        filtered = filtered[filtered["aspects_mentioned"].apply(lambda lst: any(a in lst for a in aspect_filter))]

    display_cols = [c for c in ["review_text_clean", "rating", "sentiment_label", "language"] if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(50), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ──────────────────────────────────────────────────────────────────────────
st.title("🍽️ Restaurant Review Intelligence")
st.caption("Turning raw Google reviews into an executive-style feedback report — NLP pipeline demo.")

mode = st.sidebar.radio("Mode", ["Majestic Café (Porto) — case study", "Analyze another restaurant (live)"])

if mode == "Majestic Café (Porto) — case study":
    df, aspect_summary, summary_text = load_precomputed()
    render_dashboard(df, aspect_summary, summary_text, "Majestic Café, Porto")

else:
    # Default to the app owner's keys stored in Streamlit Cloud "Secrets"
    # (Settings > Secrets, as TOML: SERPAPI_KEY = "..." / GROQ_KEY = "...")
    # so anyone using the deployed app (e.g. a professor grading it) doesn't need their own keys.
    default_serpapi_key = st.secrets.get("SERPAPI_KEY", "")
    default_groq_key = st.secrets.get("GROQ_KEY", "")

    st.sidebar.markdown("### API Keys")
    if default_serpapi_key and default_groq_key:
        st.sidebar.caption("Using the app owner's API keys by default. You can optionally use your own below.")
    else:
        st.sidebar.caption("Keys are only used for this session — never stored or logged.")

    use_own_keys = st.sidebar.checkbox("Use my own API keys instead", value=not (default_serpapi_key and default_groq_key))
    if use_own_keys:
        serpapi_key = st.sidebar.text_input("SerpApi key", type="password")
        groq_key = st.sidebar.text_input("Groq key", type="password")
    else:
        serpapi_key = default_serpapi_key
        groq_key = default_groq_key

    restaurant_query = st.text_input("Restaurant name (add city for best results)", placeholder="e.g. Cervejaria Gazela, Porto")

    if st.button("Analyze", type="primary"):
        if not serpapi_key or not groq_key:
            st.error("Please enter both API keys in the sidebar.")
        elif not restaurant_query:
            st.error("Please enter a restaurant name.")
        else:
            with st.spinner(f"Fetching and analyzing reviews for '{restaurant_query}'... this can take 30-60 seconds."):
                try:
                    result = run_live_pipeline(restaurant_query, serpapi_key, groq_key)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    result = None

            if result is None:
                st.error("Couldn't find that restaurant or no reviews were returned. Try a more specific name (add the city).")
            else:
                render_dashboard(result["df"], result["aspect_summary"], result["summary_text"], result["place_name"])
