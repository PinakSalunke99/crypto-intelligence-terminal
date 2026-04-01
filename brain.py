import ollama
from prophet import Prophet
import pandas as pd
import numpy as np
import logging

# Fault-Tolerant Imports
FINBERT_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    FINBERT_AVAILABLE = True
except Exception:
    pass

def get_llm_sentiment(text_batch):
    """
    Analyzes sentiment using local LLM with strict formatting.
    Target: 70-75% accuracy. [cite: 51]
    """
    combined_text = "\n".join(text_batch[:5])
    
    # SYSTEM PROMPT: Forces the LLM to skip talking and just give the data
    system_prompt = "You are a financial analyst. Output ONLY in this format: LABEL | SCORE | REASON. Score is 0 to 1."
    user_content = f"Analyze: {combined_text}"
    
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ])
        return response['message']['content'].strip()
    
    except Exception:
        if FINBERT_AVAILABLE:
            try:
                inputs = tokenizer(combined_text[:512], return_tensors="pt", padding=True, truncation=True)
                outputs = model(**inputs)
                prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
                scores = prediction.detach().numpy()[0]
                label = "BULLISH" if np.argmax(scores) == 0 else "BEARISH" if np.argmax(scores) == 1 else "NEUTRAL"
                return f"{label} | {np.max(scores):.2f} | Fallback: FinBERT"
            except:
                pass
        return "NEUTRAL | 0.50 | System Failsafe Triggered"

def predict_directional_movement(df):
    """
    Time-series forecasting for 1h, 4h, and 24h. [cite: 60]
    """
    try:
        df_prophet = df.rename(columns={'timestamp': 'ds', 'close': 'y'})
        m = Prophet(changepoint_prior_scale=0.05, interval_width=0.95)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=96, freq='15min')
        forecast = m.predict(future)
        
        curr = df['close'].iloc[-1]
        results = {}
        for label, step in [('1h', 4), ('4h', 16), ('24h', 96)]:
            p = forecast['yhat'].iloc[-(97-step)]
            diff = ((p - curr) / curr) * 100
            upper = forecast['yhat_upper'].iloc[-(97-step)]
            lower = forecast['yhat_lower'].iloc[-(97-step)]
            conf = 1 - ((upper - lower) / p) if p != 0 else 0
            results[label] = {
                "direction": "UP" if diff > 0.3 else "DOWN" if diff < -0.3 else "SIDEWAYS",
                "change": f"{diff:.2f}%",
                "confidence": f"{max(0, conf):.2%}"
            }
        return results
    except Exception:
        return {h: {"direction": "SIDEWAYS", "change": "0.00%", "confidence": "0.0%"} for h in ['1h', '4h', '24h']}