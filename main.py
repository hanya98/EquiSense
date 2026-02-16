"""
StockSense – Market Mood Radar
Requirements (pip install):
    streamlit requests pandas numpy yfinance feedparser
    transformers torch                  <- for FinBERT
    google-generativeai                 <- for Gemini 2.5 Flash

API keys in .streamlit/secrets.toml:
    GNEWS_API_KEY   = "..."
    GEMINI_API_KEY  = "..."
    NEWSAPI_KEY     = "..."   
    FINNHUB_KEY     = "..."   
"""

# -- Standard library ----------------------------------------------------------
import time
import re
from datetime import datetime

# -- Third-party (always available) --------------------------------------------
import streamlit as st
import requests
import pandas as pd
import numpy as np

# -- Optional: yfinance --------------------------------------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# -- Optional: feedparser ------------------------------------------------------
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# -- Optional: transformers + torch (FinBERT) ----------------------------------
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# -- Optional: google-generativeai (Gemini) ------------------------------------
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ==============================================================================
# PAGE CONFIG & STYLES
# ==============================================================================

st.set_page_config(
    page_title="StockSense - Market Mood Radar",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"]          { font-family: 'DM Sans', sans-serif; }
h1, h2, h3                          { font-family: 'DM Serif Display', serif; }
#MainMenu, footer                   { visibility: hidden; }

section[data-testid="stSidebar"]    { background: #0f1117; border-right: 1px solid #1e2130; }
section[data-testid="stSidebar"] *  { color: #e0e0e0 !important; }

[data-testid="stMetric"]            { background: #161b27; border: 1px solid #1e2a3a;
                                      border-radius: 10px; padding: 12px 16px; }
[data-testid="stMetricValue"]       { font-size: 1.4rem !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"]       { font-size: 0.75rem !important; opacity: 0.7; }

.article-card   { background: #161b27; border: 1px solid #1e2a3a; border-radius: 10px;
                  padding: 12px 16px; margin-bottom: 8px; }
.art-source     { font-size: 0.72rem; color: #6b7280; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.08em; }
.art-text       { font-size: 0.88rem; color: #d1d5db; margin-top: 4px; line-height: 1.5; }
.art-meta       { font-size: 0.72rem; color: #4b5563; margin-top: 6px; }
.badge-pos      { color: #4ade80; font-weight: 700; }
.badge-neg      { color: #f87171; font-weight: 700; }
.badge-neu      { color: #9ca3af; font-weight: 700; }

.sri-badge      { display: inline-block; padding: 4px 12px; border-radius: 20px;
                  font-size: 0.82rem; font-weight: 600; letter-spacing: 0.05em; }
.sri-high       { background: #0d3d1f; color: #4ade80; border: 1px solid #166534; }
.sri-medium     { background: #3d2e00; color: #fbbf24; border: 1px solid #92400e; }
.sri-low        { background: #3d0d0d; color: #f87171; border: 1px solid #991b1b; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTS
# ==============================================================================

FINBERT_MODEL = "ProsusAI/finbert"
GEMINI_MODEL  = "gemini-2.5-flash"

SOURCE_CREDIBILITY = {
    "Bloomberg":           1.00,
    "Reuters":             1.00,
    "Financial Times":     0.95,
    "Wall Street Journal": 0.95,
    "CNBC":                0.85,
    "Fortune":             0.80,
    "Business Insider":    0.75,
    "Finnhub":             0.70,
    "GNews":               0.65,
    "NewsAPI":             0.70,
}

# ==============================================================================
# SECRETS  (never shown in UI)
# ==============================================================================

def _secret(key):
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""

GNEWS_KEY   = _secret("GNEWS_API_KEY")
GEMINI_KEY  = _secret("GEMINI_API_KEY")
NEWSAPI_KEY = _secret("NEWSAPI_KEY")
FINNHUB_KEY = _secret("FINNHUB_KEY")

# ==============================================================================
# MODEL LOADING
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_finbert():
    """Load FinBERT. Returns pipeline or None."""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        return hf_pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            max_length=512,
            truncation=True,
        )
    except Exception as e:
        st.warning(f"FinBERT failed to load ({e}). Keyword fallback will be used.")
        return None

_finbert = load_finbert()

# ==============================================================================
# NEWS FETCHERS
# ==============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_gnews(api_key, query):
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "lang": "en", "max": 30, "apikey": api_key},
            timeout=12,
        )
        r.raise_for_status()
        articles = []
        for a in r.json().get("articles", []):
            text = a.get("title", "")
            desc = a.get("description", "")
            if desc:
                text += " " + desc
            articles.append({
                "text": text,
                "source": a.get("source", {}).get("name", "GNews"),
                "url": a.get("url", ""),
                "publishedAt": a.get("publishedAt", ""),
            })
        return articles
    except requests.HTTPError as e:
        code = e.response.status_code
        if code == 429:
            st.warning("GNews rate limit reached.")
        else:
            st.warning(f"GNews HTTP {code}.")
        return []
    except Exception as e:
        st.warning(f"GNews error: {e}")
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_newsapi(api_key, query):
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "apiKey": api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 30,
                "sources": "bloomberg,cnbc,financial-times,the-wall-street-journal,fortune,business-insider",
            },
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            return []
        articles = []
        for a in data.get("articles", []):
            text = a.get("title", "")
            desc = a.get("description", "")
            if desc:
                text += " " + desc
            articles.append({
                "text": text,
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "url": a.get("url", ""),
                "publishedAt": a.get("publishedAt", ""),
            })
        return articles
    except requests.HTTPError as e:
        code = e.response.status_code
        if code == 426:
            st.warning("NewsAPI: Upgrade required for premium sources.")
        elif code == 429:
            st.warning("NewsAPI rate limit reached.")
        return []
    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_finnhub(api_key, query):
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": api_key},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return []
        query_words = set(query.lower().split())
        articles = []
        for a in data[:30]:
            headline = a.get("headline", "")
            summary  = a.get("summary", "")
            combined = (headline + " " + summary).lower()
            if query.lower() in combined or query_words & set(combined.split()):
                ts = a.get("datetime", 0)
                articles.append({
                    "text": headline + (" " + summary if summary else ""),
                    "source": a.get("source", "Finnhub"),
                    "url": a.get("url", ""),
                    "publishedAt": datetime.fromtimestamp(ts).isoformat() if ts else "",
                })
        return articles
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            st.warning("Finnhub rate limit reached.")
        return []
    except Exception as e:
        st.warning(f"Finnhub error: {e}")
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_rss(feed_urls, source_name, query):
    if not FEEDPARSER_AVAILABLE:
        return []
    query_words = set(query.lower().split())
    articles = []
    try:
        for url in feed_urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title   = entry.get("title", "")
                summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
                combined = (title + " " + summary).lower()
                if query.lower() in combined or query_words & set(combined.split()):
                    articles.append({
                        "text": title + (" " + summary[:200] if summary else ""),
                        "source": source_name,
                        "url": entry.get("link", ""),
                        "publishedAt": entry.get("published", ""),
                    })
    except Exception:
        pass
    return articles

# ==============================================================================
# MARKET DATA
# ==============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_nifty():
    if not YFINANCE_AVAILABLE:
        return None, None
    try:
        hist = yf.Ticker("^NSEI").history(period="5d", interval="1d")
        if hist.empty:
            return None, None
        close  = float(hist["Close"].iloc[-1])
        change = float(
            (close - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
        ) if len(hist) >= 2 else 0.0
        return round(close, 2), round(change, 2)
    except Exception:
        return None, None


@st.cache_data(ttl=300, show_spinner=False)
def get_vix():
    if not YFINANCE_AVAILABLE:
        return None
    try:
        hist = yf.Ticker("^INDIAVIX").history(period="5d", interval="1d")
        return round(float(hist["Close"].iloc[-1]), 2) if not hist.empty else None
    except Exception:
        return None


def get_pcr():
    """Simulated — live NSE PCR not available via free API."""
    return round(float(np.random.uniform(0.6, 1.4)), 2)

# ==============================================================================
# SENTIMENT ANALYSIS
# ==============================================================================

_POS_WORDS = {
    "surge", "rally", "gain", "rise", "growth", "profit", "beat", "record",
    "bullish", "upgrade", "strong", "positive", "outperform", "buy", "boom",
    "recovery", "high", "up", "increased", "improved", "optimistic", "jump",
    "soar", "robust", "expand", "upside",
}
_NEG_WORDS = {
    "fall", "drop", "loss", "decline", "crash", "bearish", "downgrade", "weak",
    "negative", "sell", "bust", "recession", "low", "down", "decreased", "cut",
    "risk", "fear", "concern", "warning", "miss", "below", "contraction", "debt",
    "slump", "plunge", "tumble", "volatile", "default",
}


def _keyword_score(texts):
    """Zero-dependency fallback scorer using curated financial word lists."""
    results = []
    for text in texts:
        words = set(text.lower().split())
        p = len(words & _POS_WORDS)
        n = len(words & _NEG_WORDS)
        if p > n:
            results.append({"label": "positive", "score": round(min(0.55 + p * 0.04, 0.92), 3)})
        elif n > p:
            results.append({"label": "negative", "score": round(min(0.55 + n * 0.04, 0.92), 3)})
        else:
            results.append({"label": "neutral", "score": 0.60})
    return results


def analyze_sentiment(articles):
    """Returns (fear%, greed%, neutral%, DataFrame)."""
    if not articles:
        return 0.0, 0.0, 0.0, pd.DataFrame()

    subset = articles[:50]
    texts  = [a["text"][:512] for a in subset]

    if _finbert is not None:
        try:
            raw     = _finbert(texts)
            results = [{"label": r["label"].lower(), "score": round(r["score"], 3)} for r in raw]
        except Exception as e:
            st.warning(f"FinBERT inference failed ({e}), using keyword fallback.")
            results = _keyword_score(texts)
    else:
        results = _keyword_score(texts)

    df = pd.DataFrame({
        "text":   [a["text"][:120] + "..." for a in subset],
        "source": [a["source"]             for a in subset],
        "label":  [r["label"]              for r in results],
        "score":  [r["score"]              for r in results],
        "url":    [a.get("url", "")        for a in subset],
    })

    total   = max(len(df), 1)
    fear    = round(len(df[df["label"] == "negative"]) / total * 100, 2)
    greed   = round(len(df[df["label"] == "positive"]) / total * 100, 2)
    neutral = round(len(df[df["label"] == "neutral"])  / total * 100, 2)
    return fear, greed, neutral, df

# ==============================================================================
# ANALYTICS HELPERS
# ==============================================================================

def _recency_weight(published_at):
    try:
        if not published_at:
            return 0.5
        pub = pd.to_datetime(published_at)
        now = pd.Timestamp.now(tz=pub.tz)
        hrs = (now - pub).total_seconds() / 3600
        if hrs < 6:  return 1.0
        if hrs < 24: return 0.9
        if hrs < 48: return 0.7
        if hrs < 72: return 0.5
        return 0.3
    except Exception:
        return 0.5


def calculate_sri(df, articles):
    if df.empty:
        return 0.0, "Low"
    weights = []
    for i, row in df.iterrows():
        cred = SOURCE_CREDIBILITY.get(row["source"], 0.6)
        rec  = _recency_weight(articles[i]["publishedAt"] if i < len(articles) else "")
        conf = row["score"]
        weights.append(cred * 0.4 + rec * 0.3 + conf * 0.3)
    score = round(float(np.mean(weights)) * 100, 1)
    label = "High" if score >= 75 else "Medium" if score >= 50 else "Low"
    return score, label


def sentiment_dispersion(df):
    if df.empty:
        return 0.0
    return round(float(1 - df["label"].value_counts(normalize=True).max()), 3)


def detect_regime(vix, dispersion):
    if vix is None:
        return "Unknown", "info"
    if vix > 18 and dispersion > 0.35:
        return "Risk-Off", "danger"
    if dispersion > 0.40 and 12 < vix <= 18:
        return "Event-Driven", "warning"
    if vix < 12 and dispersion < 0.25:
        return "Risk-On", "success"
    return "Event-Driven", "info"


def detect_divergence(fear, greed, nifty_change):
    if nifty_change is None:
        return None, None
    if greed > 75 and nifty_change < -1.5:
        return "Extreme Bullish Divergence - Distribution Risk", "danger"
    if fear  > 75 and nifty_change >  1.5:
        return "Extreme Bearish Divergence - Capitulation Signal", "danger"
    if greed > 65 and nifty_change < -0.5:
        return "Bullish Sentiment / Bearish Price", "warning"
    if fear  > 65 and nifty_change >  0.5:
        return "Bearish Sentiment / Bullish Price", "warning"
    return None, None


def top_contributors(df, n=5):
    if df.empty:
        return [], []
    pos = df[df["label"] == "positive"].nlargest(n, "score").to_dict("records")
    neg = df[df["label"] == "negative"].nlargest(n, "score").to_dict("records")
    return pos, neg


def generate_signal(fear, greed, neutral, vix, pcr, nifty_change):
    if vix is None:
        return "INSUFFICIENT DATA", "info"
    s  = 0
    s += -3 if fear  > 70 else -2 if fear  > 55 else -1 if fear  > 40 else 0
    s +=  3 if greed > 70 else  2 if greed > 55 else  1 if greed > 40 else 0
    s += -2 if vix   > 20 else -1 if vix   > 15 else  2 if vix   < 10 else 1 if vix < 12 else 0
    s += -2 if pcr   > 1.3 else -1 if pcr  > 1.1 else  2 if pcr  < 0.7 else 1 if pcr < 0.9 else 0
    if nifty_change is not None:
        s += -1 if nifty_change < -2 else 1 if nifty_change > 2 else 0
    if s <= -5: return "EXTREME PANIC - Strong Reversal Potential", "danger"
    if s <= -3: return "HIGH FEAR - Cautious Buying Opportunity", "warning"
    if s <= -1: return "MODERATE FEAR - Wait and Watch", "info"
    if s <=  1: return "NEUTRAL ZONE - Market in Balance", "info"
    if s <=  3: return "MODERATE GREED - Stay Alert", "warning"
    if s <=  5: return "HIGH GREED - Consider Profit Booking", "warning"
    return "EXTREME EUPHORIA - Distribution Risk High", "danger"


def simulate_backtest(df, nifty_change):
    if df.empty or nifty_change is None:
        return None
    total   = max(len(df), 1)
    pos_pct = len(df[df["label"] == "positive"]) / total
    bias    = "bullish" if pos_pct >= 0.5 else "bearish"
    price   = "bullish" if nifty_change > 0 else "bearish"
    match   = bias == price
    acc     = round((0.58 if match else 0.42) + float(np.random.uniform(-0.05, 0.05)), 2)
    return {
        "sentiment_bias":    bias,
        "price_direction":   price,
        "directional_match": match,
        "simulated_accuracy": acc,
    }

# ==============================================================================
# GEMINI AI ANALYSIS
# ==============================================================================

def explain_gemini(api_key, signal, fear, greed, neutral, vix, pcr, articles, query):
    if not api_key or not GENAI_AVAILABLE:
        return None

    headlines = "\n".join(
        f"{i+1}. [{a['source']}] {a['text'][:200]}"
        for i, a in enumerate(articles[:15])
    )
    prompt = f"""You are a professional financial research analyst specialising in Indian markets.

User Search Term: "{query}"
Task: Analyse ONLY information directly related to "{query}".

Market Sentiment Data:
- Overall Signal : {signal}
- Fear           : {fear}%  | Greed: {greed}%  | Neutral: {neutral}%
- India VIX      : {vix if vix else 'N/A'}  | PCR: {pcr:.2f}

Recent News Headlines:
{headlines}

Provide a concise structured analysis with these four sections:

**1. Sentiment Summary**
Classify the overall news sentiment for {query} as Bullish / Bearish / Neutral with a one-sentence rationale.

**2. Key Insights**
Three bullet points explaining how current news affects {query}.

**3. Outlook**
One line only: Strongly Bullish / Mildly Bullish / Neutral / Mildly Bearish / Strongly Bearish

**4. Suggested Action**
Accumulate / Hold / Book Profits / Avoid with a brief risk note.

Rules: Focus ONLY on {query}. Be factual and data-driven. Max 250 words."""

    try:
        genai.configure(api_key=api_key)
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        if response and hasattr(response, "text") and response.text:
            return response.text
        st.warning("Gemini returned an empty response. Try a different search term.")
        return None
    except Exception as e:
        msg = str(e).lower()
        if "api_key_invalid" in msg or "invalid api key" in msg:
            st.error("Invalid Gemini API key. Check GEMINI_API_KEY in your secrets.")
        elif "quota" in msg or "resource_exhausted" in msg:
            st.warning("Gemini quota exceeded - wait 60 s and retry.")
        elif "429" in str(e) or "rate" in msg:
            st.warning("Gemini rate limit hit - wait 60 s and retry.")
        elif "blocked" in msg or "safety" in msg:
            st.warning("Gemini filtered this query. Try rephrasing the search term.")
        elif "not found" in msg or "404" in str(e):
            st.error(f"Gemini model '{GEMINI_MODEL}' not found. Ensure your API key has Gemini 2.5 access.")
        else:
            st.error(f"Gemini error: {e}")
        return None

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown("## Search Settings")

    search_query = st.text_input(
        "Target Stock / Topic",
        value="NIFTY 50",
        help="Enter a stock name, sector index, or market event.",
    )

    use_rss = st.checkbox(
        "Include CNBC & Bloomberg RSS",
        value=FEEDPARSER_AVAILABLE,
        help="Requires the feedparser package.",
    )

    st.markdown("---")
    st.markdown("## System Status")

    # Sentiment model
    if _finbert is not None:
        st.markdown("✅ **FinBERT** (ProsusAI/finbert)")
    elif TRANSFORMERS_AVAILABLE:
        st.markdown("⚠️ **FinBERT** load failed — keyword fallback active")
    else:
        st.markdown("⚠️ **Keyword fallback** — install `transformers torch` for FinBERT")

    st.markdown(f"{'✅' if YFINANCE_AVAILABLE   else '❌'} **yfinance** market data")
    st.markdown(f"{'✅' if FEEDPARSER_AVAILABLE  else '⚠️'} **feedparser** RSS")
    st.markdown(f"{'✅' if GENAI_AVAILABLE       else '❌'} **google-generativeai**")

    st.markdown("---")
    st.markdown("## API Keys")
    st.markdown(f"{'✅' if GNEWS_KEY   else '❌'} GNews")
    st.markdown(f"{'✅' if GEMINI_KEY  else '⚠️'} Gemini 2.5 Flash")
    st.markdown(f"{'✅' if NEWSAPI_KEY else '⚠️'} NewsAPI (optional)")
    st.markdown(f"{'✅' if FINNHUB_KEY else '⚠️'} Finnhub (optional)")

    st.markdown("---")
    st.caption("API keys are loaded from .streamlit/secrets.toml and never shown here.")

# ==============================================================================
# MAIN HEADER
# ==============================================================================

st.markdown("""
<h1 style='margin-bottom:0'>📊 StockSense - Market Mood Radar</h1>
<p style='color:#6b7280;margin-top:4px;font-size:0.9rem'>
Real-time sentiment · Reliability scoring · Regime detection · Divergence alerts
</p>
""", unsafe_allow_html=True)
st.divider()

# ==============================================================================
# ANALYSE BUTTON
# ==============================================================================

if st.button("🔄 Analyse Market Sentiment", type="primary", use_container_width=True):

    if not GNEWS_KEY:
        st.error("GNews API key not configured. Add GNEWS_API_KEY to .streamlit/secrets.toml")
        st.stop()

    with st.spinner(f"Analysing '{search_query}' ..."):
        pbar   = st.progress(0)
        status = st.empty()

        # Fetch articles
        all_articles = []

        status.text("Fetching GNews ...")
        pbar.progress(10)
        all_articles += fetch_gnews(GNEWS_KEY, search_query)

        if NEWSAPI_KEY:
            status.text("Fetching NewsAPI ...")
            pbar.progress(25)
            all_articles += fetch_newsapi(NEWSAPI_KEY, search_query)

        if FINNHUB_KEY:
            status.text("Fetching Finnhub ...")
            pbar.progress(38)
            all_articles += fetch_finnhub(FINNHUB_KEY, search_query)

        if use_rss:
            status.text("Fetching CNBC RSS ...")
            pbar.progress(48)
            all_articles += fetch_rss(
                [
                    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
                    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
                ],
                "CNBC", search_query,
            )
            status.text("Fetching Bloomberg RSS ...")
            pbar.progress(55)
            all_articles += fetch_rss(
                [
                    "https://feeds.bloomberg.com/markets/news.rss",
                    "https://feeds.bloomberg.com/economics/news.rss",
                ],
                "Bloomberg", search_query,
            )

        if not all_articles:
            pbar.empty()
            status.empty()
            st.error(f"No articles found for '{search_query}'. Check your GNews key or try a different term.")
            st.stop()

        # De-duplicate by URL / text prefix
        seen = set()
        unique = []
        for a in all_articles:
            url = a.get("url", "")
            key = url if url else a["text"][:80]
            if key not in seen:
                seen.add(key)
                unique.append(a)
        all_articles = unique

        # Sentiment analysis
        backend_label = "FinBERT" if _finbert else "keyword fallback"
        status.text(f"Analysing sentiment ({backend_label}) ...")
        pbar.progress(65)
        fear, greed, neutral, sentiment_df = analyze_sentiment(all_articles)

        # Market data
        status.text("Fetching market indicators ...")
        pbar.progress(75)
        nifty_price, nifty_change = get_nifty()
        vix_value = get_vix()
        pcr_value = get_pcr()

        # Advanced metrics
        status.text("Computing reliability and regime ...")
        pbar.progress(83)
        sri_score, sri_label       = calculate_sri(sentiment_df, all_articles)
        disp                       = sentiment_dispersion(sentiment_df)
        regime, regime_color       = detect_regime(vix_value, disp)
        div_msg, div_color         = detect_divergence(fear, greed, nifty_change)
        top_pos, top_neg           = top_contributors(sentiment_df)
        bt                         = simulate_backtest(sentiment_df, nifty_change)
        signal, signal_type        = generate_signal(
            fear, greed, neutral, vix_value, pcr_value, nifty_change
        )

        # Gemini AI
        gemini_text = None
        if GEMINI_KEY and GENAI_AVAILABLE:
            status.text("Generating Gemini 2.5 Flash insights ...")
            pbar.progress(93)
            gemini_text = explain_gemini(
                GEMINI_KEY, signal, fear, greed, neutral,
                vix_value, pcr_value, all_articles, search_query,
            )

        pbar.progress(100)
        status.text("Done!")
        time.sleep(0.4)
        pbar.empty()
        status.empty()

        # Source counts
        src_counts = {}
        for a in all_articles:
            src_counts[a["source"]] = src_counts.get(a["source"], 0) + 1

        # Persist to session state
        st.session_state.update({
            "query":        search_query,
            "fear":         fear,
            "greed":        greed,
            "neutral":      neutral,
            "sentiment_df": sentiment_df,
            "all_articles": all_articles,
            "src_counts":   src_counts,
            "nifty_price":  nifty_price,
            "nifty_change": nifty_change,
            "vix":          vix_value,
            "pcr":          pcr_value,
            "signal":       signal,
            "signal_type":  signal_type,
            "gemini_text":  gemini_text,
            "sri_score":    sri_score,
            "sri_label":    sri_label,
            "dispersion":   disp,
            "regime":       regime,
            "regime_color": regime_color,
            "div_msg":      div_msg,
            "div_color":    div_color,
            "top_pos":      top_pos,
            "top_neg":      top_neg,
            "bt":           bt,
            "backend":      backend_label,
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

# ==============================================================================
# RESULTS DISPLAY
# ==============================================================================

if "fear" not in st.session_state:
    st.info("Select a topic in the sidebar and click Analyse Market Sentiment to begin.")
    st.stop()

S = st.session_state

st.markdown(f"### Results: {S.query}")
st.caption(
    f"Last updated: {S.timestamp}  |  "
    f"{len(S.all_articles)} unique articles  |  "
    f"Sentiment engine: {S.backend}"
)

# Row 1: Core sentiment metrics
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Fear",     f"{S.fear}%")
with c2: st.metric("Greed",    f"{S.greed}%")
with c3: st.metric("Neutral",  f"{S.neutral}%")
with c4: st.metric("Articles", len(S.all_articles))
with c5:
    if S.nifty_price:
        delta = f"{S.nifty_change:+.2f}%" if S.nifty_change is not None else None
        st.metric("NIFTY 50", f"{S.nifty_price:,.0f}", delta)
    else:
        st.metric("NIFTY 50", "Unavailable")

st.divider()

# Row 2: SRI / Regime / Indicators
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**Sentiment Reliability Index**")
    st.metric("SRI Score", f"{S.sri_score}/100",
              help="Weighted by source credibility, article recency and model confidence")
    cls = {"High": "sri-high", "Medium": "sri-medium", "Low": "sri-low"}.get(S.sri_label, "sri-low")
    st.markdown(f'<span class="sri-badge {cls}">{S.sri_label} Confidence</span>',
                unsafe_allow_html=True)

with c2:
    st.markdown("**Market Regime**")
    fn = {"success": st.success, "danger": st.error, "warning": st.warning}.get(
        S.regime_color, st.info
    )
    fn(f"**{S.regime}**")
    st.metric("Sentiment Dispersion", f"{S.dispersion:.2f}",
              help="Higher = more disagreement among sources")

with c3:
    st.markdown("**India VIX**")
    ind = "🔴" if S.vix and S.vix > 15 else "🟢"
    st.metric(f"{ind} Volatility Index", str(S.vix) if S.vix else "N/A")

with c4:
    st.markdown("**Put-Call Ratio** *(simulated)*")
    ind = "🔴" if S.pcr > 1.2 else "🟢" if S.pcr < 0.8 else "🟡"
    st.metric(f"{ind} PCR", f"{S.pcr:.2f}")

st.divider()

# Divergence alert
if S.div_msg:
    fn = st.error if S.div_color == "danger" else st.warning
    fn(f"**Smart Money Divergence Detected:** {S.div_msg}")
    st.caption("Divergences may signal institutional positioning different from retail sentiment.")
    st.divider()

# Signal
st.markdown("### Market Signal")
fn = {"danger": st.error, "warning": st.warning}.get(S.signal_type, st.info)
fn(S.signal)
st.divider()

# Gemini AI analysis
st.markdown(f"### Gemini 2.5 Flash - AI Financial Analysis")
if S.gemini_text:
    st.markdown(S.gemini_text)
elif not GENAI_AVAILABLE:
    st.info("Install google-generativeai and add GEMINI_API_KEY to secrets for AI analysis.")
elif not GEMINI_KEY:
    st.info("Add GEMINI_API_KEY to .streamlit/secrets.toml to enable Gemini analysis.")
else:
    st.info("Gemini analysis unavailable - check warnings above.")
st.divider()

# Top contributors
st.markdown("### Explainability - Top Signal Contributors")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Top Positive Signals**")
    if S.top_pos:
        for item in S.top_pos:
            st.markdown(
                f'<div class="article-card">'
                f'<div class="art-source">{item["source"]}</div>'
                f'<div class="art-text">{item["text"]}</div>'
                f'<div class="art-meta">'
                f'<span class="badge-pos">POSITIVE</span> confidence {item["score"]:.2f}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No strong positive signals found.")

with c2:
    st.markdown("**Top Negative Signals**")
    if S.top_neg:
        for item in S.top_neg:
            st.markdown(
                f'<div class="article-card">'
                f'<div class="art-source">{item["source"]}</div>'
                f'<div class="art-text">{item["text"]}</div>'
                f'<div class="art-meta">'
                f'<span class="badge-neg">NEGATIVE</span> confidence {item["score"]:.2f}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No strong negative signals found.")

st.divider()

# Directional alignment
st.markdown("### Directional Alignment Check")
if S.bt:
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Sentiment Bias",  S.bt["sentiment_bias"].capitalize())
    with c2: st.metric("Price Direction", S.bt["price_direction"].capitalize())
    with c3:
        icon = "✅" if S.bt["directional_match"] else "❌"
        st.metric("Match", f"{icon} {'Yes' if S.bt['directional_match'] else 'No'}")
    st.info(f"Simulated historical accuracy: **{S.bt['simulated_accuracy']*100:.0f}%** (illustrative only)")
    st.caption("Simplified check only - not a trading signal. No costs or slippage modelled.")
else:
    st.info("Insufficient data for alignment check.")
st.divider()

# Sentiment chart
st.markdown("### Sentiment Distribution")
chart_df = pd.DataFrame({
    "Sentiment":  ["Fear", "Neutral", "Greed"],
    "Percentage": [S.fear, S.neutral, S.greed],
}).set_index("Sentiment")
st.bar_chart(chart_df)
st.divider()

# Source breakdown
st.markdown("### Sources Breakdown")
cols = st.columns(max(len(S.src_counts), 1))
for col, (src, cnt) in zip(cols, S.src_counts.items()):
    with col:
        st.metric(src, cnt)
st.divider()

# Detail tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Articles", "Sentiment Data", "Source Links", "Limitations"
])

with tab1:
    rows = [{
        "Source":    a["source"],
        "Headline":  a["text"][:160] + ("..." if len(a["text"]) > 160 else ""),
        "Published": a.get("publishedAt", "N/A"),
    } for a in S.all_articles]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)

with tab2:
    if not S.sentiment_df.empty:
        st.dataframe(
            S.sentiment_df[["source", "label", "score", "text"]],
            use_container_width=True, height=420,
        )
    else:
        st.info("No sentiment data available.")

with tab3:
    for a in S.all_articles[:30]:
        url = a.get("url", "")
        if url:
            label = a["text"][:100] + "..."
            st.markdown(f"**[{a['source']}]** [{label}]({url})")

with tab4:
    st.markdown("""
### Known Limitations and Assumptions

**Model**
- FinBERT may misclassify domain-specific jargon or sarcasm.
- Headline-only analysis misses nuance in full article bodies.
- Keyword fallback has significantly lower accuracy than FinBERT.

**Data**
- RSS feeds may include stale or duplicate content (de-duplicated by URL).
- API rate limits can reduce sample size and introduce bias.
- Source credibility weights are heuristic, not empirically validated.

**Market Indicators**
- PCR is simulated — real-time NSE feed is not available via free API.
- NIFTY / VIX data depend on exchange hours and yfinance availability.

**Backtesting**
- Simulated accuracy is illustrative, not predictive.
- No transaction costs, slippage, or market impact are modelled.

**Regulatory Disclaimer**

This tool is for educational and research purposes only. It is not financial advice,
an investment recommendation, or a trading signal. Past patterns do not guarantee
future performance. Always consult a registered financial advisor before making
any investment decisions.
""")

# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption(
    "Educational purposes only · Not financial advice · "
    "Sentiment signals are probabilistic, not deterministic."
)
st.caption(
    f"Built with Streamlit · FinBERT (ProsusAI) · Gemini 2.5 Flash ({GEMINI_MODEL}) · "
    "GNews · NewsAPI · Finnhub"
)
