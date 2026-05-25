import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from pymongo import MongoClient

class DataProvider:
    """
    Class for Stage 1 & 2: Data Collection & Storage.
    Handles fetching data from yfinance and storing it in MongoDB.
    """
    
    def __init__(self, db_name: str = "apple_stock_db", collection_name: str = "historical_data"):
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        print(f"[*] Connected to MongoDB at {self.mongo_uri}, Database: {db_name}")

    def fetch_stock_data(self, ticker: str, start: str, end: str = None) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
        
        print(f"[*] Fetching data for {ticker} from {start} to {end}...")
        try:
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns.values]
            
            # Reset index to have 'Date' as a column
            data = data.reset_index()
            return data
        except Exception as e:
            print(f"[!] Error fetching data: {e}")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, collection_name: str = None) -> bool:
        """
        Save DataFrame to MongoDB.
        """
        target_collection = self.db[collection_name] if collection_name else self.collection
        print(f"[*] Saving {len(df)} records to MongoDB collection '{target_collection.name}'...")
        try:
            records = df.to_dict('records')
            # Clear existing data to avoid duplicates
            target_collection.delete_many({})
            target_collection.insert_many(records)
            print(f"[+] Data saved successfully to collection '{target_collection.name}'.")
            return True
        except Exception as e:
            print(f"[!] Error saving data to MongoDB: {e}")
            return False

    def load_data(self, collection_name: str = None) -> pd.DataFrame:
        """
        Load data from MongoDB.
        """
        target_collection = self.db[collection_name] if collection_name else self.collection
        print(f"[*] Loading data from MongoDB collection '{target_collection.name}'...")
        try:
            cursor = target_collection.find({}, {'_id': 0})
            df = pd.DataFrame(list(cursor))
            if df.empty:
                print(f"[!] No data found in collection '{target_collection.name}'.")
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"[!] Error loading data from MongoDB: {e}")
            return pd.DataFrame()
