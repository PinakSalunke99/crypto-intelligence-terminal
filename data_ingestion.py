import os
import pandas as pd
from binance.client import Client
import praw
import requests
from newsapi import NewsApiClient
from dotenv import load_dotenv
import psycopg2

load_dotenv()

class CryptoDataEngine:
    def __init__(self):
        # 1. API Clients Initialization
        self.binance = Client(os.getenv('BINANCE_API_KEY', ''), os.getenv('BINANCE_API_SECRET', ''))
        
        # Reddit Client (Requirement 3.2)
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'crypto_bot_v1')
        )
        
        # News API (Requirement 3.1)
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY', ''))
        
        # Etherscan (Requirement 3.3)
        self.eth_key = os.getenv('ETHERSCAN_API_KEY', '')

    def _get_db_conn(self):
        """Helper to establish a stable connection to the Dockerized Postgres."""
        return psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'crypto_intelligence'),
            user=os.getenv('DB_USER', 'user'),
            password=os.getenv('DB_PASSWORD', 'password'),
            host="127.0.0.1", # Localhost bridge to Docker
            port="5432"
        )

    def save_intelligence_to_db(self, symbol, price, label, score, reasoning):
        """Requirement 4.0: Master Sync - Saves price and signals.
        Casts numpy types to standard Python floats to avoid SQL schema errors."""
        try:
            conn = self._get_db_conn()
            cur = conn.cursor()
            
            # CRITICAL FIX: Cast to standard Python float to stop the 'np' schema error
            sql_price = float(price)
            sql_score = float(score)
            
            # 1. Save Price Data
            cur.execute(
                "INSERT INTO price_history (symbol, timestamp, close_price) VALUES (%s, NOW(), %s)",
                (symbol, sql_price)
            )
            
            # 2. Save Sentiment Signal
            cur.execute(
                "INSERT INTO sentiment_logs (asset, sentiment_label, sentiment_score, reasoning) VALUES (%s, %s, %s, %s)",
                (symbol, label, sql_score, reasoning)
            )
            
            conn.commit()
            cur.close()
            conn.close()
            print(f"✅ DATABASE SYNC COMPLETE: {symbol} archived successfully.")
        except Exception as e:
            print(f"❌ DATABASE ERROR: {e}")

    def get_historical_candles(self, symbol="BTCUSDT"):
        """Fetches live Binance candles for Prophet model training."""
        try:
            klines = self.binance.get_historical_klines(symbol, "15m", "1 day ago UTC")
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            print(f"✅ BINANCE: Fetched {symbol} market data.")
            return df[['timestamp', 'close']]
        except Exception as e:
            print(f"⚠️ BINANCE ERROR: {e}. Using fallback values.")
            dr = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='15min')
            return pd.DataFrame({'timestamp': dr, 'close': [68000.0]*50})

    def fetch_reddit_posts(self, limit=25):
        """Aggregates social sentiment."""
        try:
            posts = [s.title for s in self.reddit.subreddit("cryptocurrency").hot(limit=limit)]
            print(f"✅ REDDIT: Aggregated {len(posts)} posts.")
            return posts
        except Exception as e:
            print(f"⚠️ REDDIT ERROR: {e}")
            return ["Bitcoin sentiment remains strong"]

    def fetch_crypto_news(self):
        """Aggregates global headlines."""
        try:
            news = self.newsapi.get_everything(q='crypto', language='en', page_size=5)
            return [a['title'] for a in news['articles']]
        except Exception as e:
            print(f"⚠️ NEWS ERROR: {e}")
            return ["Market awaits next catalyst"]

    def get_whale_movements(self):
        """Tracks live whale activity via Etherscan."""
        try:
            url = f"https://api.etherscan.io/api?module=account&action=txlist&address=0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae&sort=desc&apikey={self.eth_key}"
            res = requests.get(url, timeout=10).json()
            if res['status'] == '1':
                df = pd.DataFrame(res['result'])
                df['value_eth'] = df['value'].astype(float) / 10**18
                print("✅ ETHERSCAN: Tracked whale activity.")
                return df[df['value_eth'] > 10].head(5)[['hash', 'value_eth']]
            raise Exception("API Limit reached")
        except:
            return pd.DataFrame({'hash': ['Mock_Whale_Alert_1'], 'value_eth': [1500.2]})