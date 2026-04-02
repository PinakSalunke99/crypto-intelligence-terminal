import streamlit as st
import pandas as pd
import numpy as np
from plyer import notification
from data_ingestion import CryptoDataEngine
from brain import get_llm_sentiment, predict_directional_movement
from backtester import BacktestEngine

# --- Safety Helpers ---
def safe_parse_sentiment(raw_text):
    try:
        parts = raw_text.split("|")
        return parts[0].strip().upper(), float(parts[1].strip())
    except:
        label = "BULLISH" if "BULL" in raw_text.upper() else "BEARISH" if "BEAR" in raw_text.upper() else "NEUTRAL"
        return label, 0.5

def trigger_alert(asset, signal, confidence):
    try:
        notification.notify(
            title=f"🚀 Signal: {asset}",
            message=f"Action: {signal} | Conf: {confidence}",
            app_name="Crypto Terminal",
            timeout=10
        )
    except: pass

# --- Configuration ---
st.set_page_config(page_title="Crypto Pulse Terminal", layout="wide")
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

engine = CryptoDataEngine()

# --- Header ---
st.title("🛡️ Self-Hosted Crypto Intelligence Terminal")
st.caption("NMIMS INNOVATHON 2026 | Local-First AI Intelligence Terminal")

# --- Sentiment Heatmap ---
coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
h_cols = st.columns(len(coins))
for i, coin in enumerate(coins):
    score = np.random.uniform(0.3, 0.8)
    color = "green" if score > 0.6 else "red" if score < 0.4 else "gray"
    h_cols[i].markdown(f"<div style='background-color:{color}; padding:20px; text-align:center; border-radius:10px; color:white;'><b>{coin}</b><br>{score:.2f}</div>", unsafe_allow_html=True)

st.divider()

# --- Execution ---
selected_asset = st.selectbox("Deep Dive Analysis", coins)

if st.button("🚀 Generate Intelligence & Signals"):
    with st.spinner(f"Running ML Pipeline for {selected_asset}..."):
        # A. Pull Data (Using Scraper for FA)
        price_df = engine.get_historical_candles(selected_asset)
        posts = engine.fetch_twitter_posts(query=selected_asset)
        news = engine.fetch_crypto_news() # Deep Scraper triggered here
        whales = engine.get_whale_movements()
        
        # B. AI Brain Processing (Mistral + Prophet)
        raw_sentiment = get_llm_sentiment(posts + news)
        preds = predict_directional_movement(price_df)
        label, score = safe_parse_sentiment(raw_sentiment)
        
        # C. Synchronized Timing
        execution_time = pd.Timestamp.now()
        
        # --- UI DISPLAY ---
        col_main, col_side = st.columns([2, 1])
        with col_main:
            st.write(f"### Sentiment: {label} ({score})")
            st.info(f"**AI Reasoning:** {raw_sentiment}")
            
            st.write("#### 🎯 Price Targets & Lag Range (±0.5%)")
            t_cols = st.columns(3)
            for i, timeframe in enumerate(['1h', '4h', '24h']):
                with t_cols[i]:
                    st.metric(f"{timeframe} Target", f"${preds[timeframe]['price']}", preds[timeframe]['change'])
                    st.caption(f"**Range:** {preds[timeframe]['range']}")
        
        with col_side:
            bt = BacktestEngine()
            stats = bt.run(price_df, score, preds)
            st.subheader("📊 30-Day Simulation")
            st.metric("Win Rate", stats['win_rate'])
            st.metric("Sharpe Ratio", stats['sharpe_ratio'])
            
            st.write("#### 🐋 Whale Activity")
            st.dataframe(whales[['hash', 'value_eth']], width='stretch')

        # --- DATABASE SYNC ---
        current_price = price_df.iloc[-1]['close']
        engine.save_intelligence_to_db(selected_asset, current_price, label, score, raw_sentiment)
        
        # Signal Logic
        sig = "BUY" if ("BULL" in label and preds['4h']['direction'] == "UP") else "SELL" if ("BEAR" in label and preds['4h']['direction'] == "DOWN") else "HOLD"
        if sig != "HOLD": trigger_alert(selected_asset, sig, preds['4h']['confidence'])

        # Update History Log
        st.session_state.signal_log.insert(0, {
            "Time": execution_time.strftime("%H:%M:%S"),
            "Asset": selected_asset,
            "Signal": sig,
            "Reason": f"{label} + {preds['4h']['direction']} Trend"
        })

# --- Live Log ---
st.divider()
st.subheader("📜 Live Signal Log")
if st.session_state.signal_log:
    st.table(pd.DataFrame(st.session_state.signal_log).head(10))