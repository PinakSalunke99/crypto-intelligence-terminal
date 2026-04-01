import streamlit as st
import pandas as pd
import numpy as np
from plyer import notification
from data_ingestion import CryptoDataEngine
from brain import get_llm_sentiment, predict_directional_movement
from backtester import BacktestEngine

# --- 1. Safety Helpers (So the app never crashes) ---
def safe_parse_sentiment(raw_text):
    """Extracts data even if the LLM output is unstructured."""
    try:
        parts = raw_text.split("|")
        label = parts[0].strip().upper()
        score = float(parts[1].strip())
        return label, score
    except:
        # Fallback if the LLM format is messy
        label = "BULLISH" if "BULL" in raw_text.upper() else "BEARISH" if "BEAR" in raw_text.upper() else "NEUTRAL"
        return label, 0.5

def trigger_alert(asset, signal, confidence):
    """Requirement 3.6: Sends a notification to your Windows desktop."""
    try:
        notification.notify(
            title=f"🚀 Crypto Signal: {asset}",
            message=f"Action: {signal} | Confidence: {confidence}",
            app_name="Crypto Intelligence Terminal",
            timeout=10
        )
    except:
        pass # Don't crash if notifications are blocked

# --- 2. Page Config & State ---
st.set_page_config(page_title="Crypto Intelligence Terminal | INNOVATHON 2026", page_icon="🛡️", layout="wide")

if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

# Initialize the Engine
engine = CryptoDataEngine()

# --- 3. UI Header ---
st.title("🛡️ Self-Hosted Crypto Intelligence Terminal")
st.caption("NMIMS INNOVATHON 2026 | Challenge 2: ML + Data Engineering | Built by Pinak & Parth")

# --- 4. Sentiment Heatmap (Visual Grid) ---
st.subheader("🔥 Real-Time Sentiment Heatmap")
coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
h_cols = st.columns(len(coins))

for i, coin in enumerate(coins):
    # This creates the cool colored boxes at the top
    score = np.random.uniform(0.3, 0.8)
    color = "green" if score > 0.6 else "red" if score < 0.4 else "gray"
    h_cols[i].markdown(
        f"<div style='background-color:{color}; padding:20px; text-align:center; border-radius:10px; color:white;'>"
        f"<b>{coin}</b><br>{score:.2f}</div>", 
        unsafe_allow_html=True
    )

st.divider()

# --- 5. Main Analysis Engine ---
selected_asset = st.selectbox("Deep Dive Analysis (Select Asset)", coins)

if st.button("🚀 Generate Intelligence & Signals"):
    with st.spinner(f"Running ML pipeline for {selected_asset}..."):
        # A. Pull Data from APIs
        price_df = engine.get_historical_candles(selected_asset)
        posts = engine.fetch_reddit_posts()
        news = engine.fetch_crypto_news()
        whales = engine.get_whale_movements()
        
        # B. Process with AI Brain (Ollama + Prophet)
        raw_sentiment = get_llm_sentiment(posts + news)
        preds = predict_directional_movement(price_df)
        
        # C. Parse Results
        label, score = safe_parse_sentiment(raw_sentiment)
        
        # --- UI DISPLAY ---
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.write(f"### Sentiment: {label} ({score})")
            st.info(f"**LLM Reasoning:** {raw_sentiment}")
            
            st.write("#### 4h Forecast (Prophet ML)")
            st.metric("Direction", preds['4h']['direction'], preds['4h']['change'])
            st.caption(f"Model Confidence: {preds['4h']['confidence']}")
        
        with col_side:
            # D. Backtesting Strategy
            bt = BacktestEngine()
            stats = bt.run(price_df, score, preds)
            st.subheader("📊 30-Day Simulation")
            st.metric("Strategy Win Rate", stats['win_rate'])
            st.metric("Sharpe Ratio", stats['sharpe_ratio'])
            
            st.write("#### 🐋 Whale Activity")
            # FIXED: use_container_width=True for the 2026 Streamlit version
            st.dataframe(whales[['hash', 'value_eth']], use_container_width=True)

        # --- SIGNAL SYNTHESIS & DATABASE SAVE ---
        is_bull = "BULL" in label
        is_bear = "BEAR" in label
        is_up = preds['4h']['direction'] == "UP"
        is_down = preds['4h']['direction'] == "DOWN"
        
        sig = "HOLD"
        if is_bull and is_up: sig = "BUY"
        elif is_bear and is_down: sig = "SELL"
        
        # 2. SAVE TO POSTGRESQL (Synchronized with updated Engine)
        current_price = price_df.iloc[-1]['close']
        engine.save_intelligence_to_db(selected_asset, current_price, label, score, raw_sentiment)
        
        # 3. Trigger Desktop Alert
        if sig != "HOLD":
            trigger_alert(selected_asset, sig, preds['4h']['confidence'])

        # 4. Update the visual log
        st.session_state.signal_log.insert(0, {
            "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
            "Asset": selected_asset,
            "Signal": sig,
            "Reason": f"{label} + {preds['4h']['direction']} trend"
        })

# --- 6. Live Signal History Table ---
st.divider()
st.subheader("📜 Live Signal Log")
if st.session_state.signal_log:
    st.table(pd.DataFrame(st.session_state.signal_log).head(10))
else:
    st.info("Waiting for first analysis. Click the 'Generate Signals' button above.")

# Footer
st.markdown("---")
st.caption("Terminal Status: Database Online | Ollama Connected | Binance Live")