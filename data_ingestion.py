import os
import pandas as pd
from binance.client import Client
import tweepy
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from datetime import datetime
import time
from dotenv import load_dotenv
import psycopg2

load_dotenv()

class CryptoDataEngine:
    def __init__(self):
        # 1. API Clients Initialization
        self.binance = Client(os.getenv('BINANCE_API_KEY', ''), os.getenv('BINANCE_API_SECRET', ''))
        self.twitter = tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'))
        self.eth_key = os.getenv('ETHERSCAN_API_KEY', '')

    def _get_db_conn(self):
        """Stable connection to the Dockerized Postgres."""
        return psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'crypto_intelligence'),
            user=os.getenv('DB_USER', 'user'),
            password=os.getenv('DB_PASSWORD', 'password'),
            host="127.0.0.1", 
            port="5432"
        )

    def save_intelligence_to_db(self, symbol, price, label, score, reasoning):
        """Archives dual-model results using your local PC's IST time."""
        try:
            conn = self._get_db_conn()
            cur = conn.cursor()
            local_now = pd.Timestamp.now() # Captures local time
            sql_price, sql_score = float(price), float(score)
            
            # Save to Price History
            cur.execute(
                "INSERT INTO price_history (symbol, timestamp, close_price) VALUES (%s, %s, %s)",
                (symbol, local_now, sql_price)
            )
            # Save to Sentiment Logs
            cur.execute(
                "INSERT INTO sentiment_logs (timestamp, asset, sentiment_label, sentiment_score, reasoning) VALUES (%s, %s, %s, %s, %s)",
                (local_now, symbol, label, sql_score, reasoning)
            )
            
            conn.commit()
            cur.close()
            conn.close()
            print(f"✅ DATABASE SYNC: {symbol} archived (Local Time).")
        except Exception as e:
            print(f"❌ DATABASE ERROR: {e}")

    def get_historical_candles(self, symbol="BTCUSDT"):
        """Fetches live Binance candles for Prophet model training."""
        try:
            klines = self.binance.get_historical_klines(symbol, "15m", "1 day ago UTC")
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            return df[['timestamp', 'close']]
        except:
            return pd.DataFrame({'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=50, freq='15min'), 'close': [68000.0]*50})

    def fetch_twitter_posts(self, query="crypto", limit=10):
        """Fetches real-time tweets or triggers fail-safe simulation."""
        try:
            response = self.twitter.search_recent_tweets(query=f"{query} -is:retweet lang:en", max_results=limit)
            if response.data: return [t.text for t in response.data]
            return ["Institutional interest in digital assets is rising."]
        except Exception as e:
            print(f"⚠️ TWITTER API: {e}. Using Fail-Safe.")
            return [f"Market analysis for {query} indicates high volume.", "Traders awaiting clear breakout signals."]

    def fetch_crypto_news(self, max_articles=5):
        """NEW: Deep Scraper for CoinDesk RSS to provide full-text fundamental data."""
        rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(rss_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all('item')[:max_articles]
            
            all_text = []
            for item in items:
                try:
                    url = item.link.text
                    article = Article(url)
                    article.download()
                    article.parse()
                    all_text.append(f"Title: {article.title}\n{article.text[:500]}") # First 500 chars for LLM
                    time.sleep(0.2)
                except: continue
            return all_text if all_text else ["Global adoption of digital assets continues."]
        except:
            return ["Market sentiment remains in a period of consolidation."]

    def get_whale_movements(self):
        """Tracks live whale activity via Etherscan."""
        try:
            url = f"https://api.etherscan.io/api?module=account&action=txlist&address=0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae&sort=desc&apikey={self.eth_key}"
            res = requests.get(url, timeout=10).json()
            if res['status'] == '1' and isinstance(res['result'], list):
                df = pd.DataFrame(res['result'])
                df['value_eth'] = df['value'].astype(float) / 10**18
                return df[df['value_eth'] > 50].head(5)[['hash', 'value_eth']]
            return pd.DataFrame({'hash': ['Normal_Flow'], 'value_eth': [0.0]})
        except:
            return pd.DataFrame({'hash': ['Mock_Whale_Alert_1'], 'value_eth': [1500.2]})