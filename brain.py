import ollama
from prophet import Prophet
import pandas as pd
import numpy as np

def get_llm_sentiment(text_batch):
    """Analyzes deep-scraped content via local Mistral 7B."""
    combined_text = "\n".join(text_batch[:5])
    system_prompt = "You are a financial analyst. Output ONLY in this format: LABEL | SCORE | REASON."
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Analyze: {combined_text}"}
        ])
        return response['message']['content'].strip()
    except:
        return "NEUTRAL | 0.50 | System Failsafe Triggered"

def predict_directional_movement(df):
    """Time-series forecasting with specific Lag Range calculation."""
    try:
        df_prophet = df.rename(columns={'timestamp': 'ds', 'close': 'y'})
        m = Prophet(changepoint_prior_scale=0.05, interval_width=0.95)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=96, freq='15min')
        forecast = m.predict(future)
        
        curr = df['close'].iloc[-1]
        results = {}
        for label, step in [('1h', 4), ('4h', 16), ('24h', 96)]:
            predicted_price = float(forecast['yhat'].iloc[-(97-step)])
            
            # LAG RANGE Calculation (±0.5%)
            range_lower = predicted_price * 0.995
            range_upper = predicted_price * 1.005
            
            diff = ((predicted_price - curr) / curr) * 100
            conf = 1 - ((forecast['yhat_upper'].iloc[-(97-step)] - forecast['yhat_lower'].iloc[-(97-step)]) / predicted_price)
            
            results[label] = {
                "direction": "UP" if diff > 0.3 else "DOWN" if diff < -0.3 else "SIDEWAYS",
                "price": round(predicted_price, 2),
                "range": f"${round(range_lower, 2)} - ${round(range_upper, 2)}",
                "change": f"{diff:.2f}%",
                "confidence": f"{max(0, conf):.2%}"
            }
        return results
    except:
        return {h: {"direction": "SIDEWAYS", "price": 0, "range": "N/A", "change": "0.00%", "confidence": "0%"} for h in ['1h', '4h', '24h']}