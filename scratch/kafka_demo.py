import time
import json
import random
from datetime import datetime
import os
import threading
from kafka import KafkaProducer, KafkaConsumer

# Directory acting as our simulated Hadoop HDFS / Data Lake
DATA_LAKE_DIR = "data/data_lake"
os.makedirs(DATA_LAKE_DIR, exist_ok=True)

# ----------------- ENVIRONMENT LOADER -----------------
def load_env():
    """Manually parse .env file to load configuration variables into os.environ."""
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # Clean whitespaces and quotes
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    os.environ[key] = val
                    
# Load configuration
load_env()

# ----------------- SECTION 1: REAL KAFKA CONFIGURATION -----------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Khởi tạo Kafka Producer toàn cục (sẽ kết nối tới Kafka )
try:
    print(f"[Producer] Connecting to Kafka Broker at {KAFKA_BOOTSTRAP_SERVERS}...")
    kafka_producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3
    )
    print("[Producer] Kafka Producer initialized successfully.")
except Exception as e:
    print(f"[Producer] CRITICAL: Failed to initialize Kafka Producer: {e}")
    print("[Producer] Please ensure Docker container is running (docker-compose up -d).")
    kafka_producer = None

# ----------------- SECTION 2: KAFKA PRODUCER (Alpaca WebSocket / Fallback Simulation) -----------------
class StockProducer(threading.Thread):
    def __init__(self, ticker: str = "AAPL"):
        super().__init__()
        self.ticker = ticker
        self.running = True
        self.alpaca_stream = None
        self.event_loop = None

    def get_latest_mongodb_price(self) -> float:
        """Fetch the last real closing price of AAPL from MongoDB as a backup simulation base."""
        try:
            from pymongo import MongoClient
            client = MongoClient("mongodb://localhost:27017/")
            db = client["apple_stock_db"]
            coll = db["aapl_raw"]
            # Find the most recent record sorted by Date descending
            latest_record = list(coll.find().sort([("Date", -1), ("date", -1)]).limit(1))
            if latest_record:
                price = float(latest_record[0].get("Close") or latest_record[0].get("close") or 312.0)
                print(f"[Producer] Loaded latest price {price} USD from MongoDB as simulation base.")
                return price
        except Exception as e:
            print(f"[Producer] Could not read latest price from MongoDB: {e}")
        return 312.0

    def run_yfinance_backfill(self):
        print(f"[Producer] Attempting to backfill historical data for {self.ticker} using yfinance...")
        try:
            import yfinance as yf
            import pandas as pd
            # Tải dữ liệu 5 ngày gần nhất, interval 1 giờ
            data = yf.download(self.ticker, period="5d", interval="1h")
            if data.empty:
                print("[Producer] yfinance returned empty dataset.")
                return
                
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns.values]
                
            data = data.reset_index()
            
            print(f"[Producer] Fetched {len(data)} hourly bars from yfinance. Sending to Kafka...")
            for _, row in data.iterrows():
                # Extract date and time
                dt = row.get('Datetime') or row.get('Date')
                dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f") if hasattr(dt, 'strftime') else str(dt)
                
                event_data = {
                    "ticker": self.ticker,
                    "timestamp": dt_str,
                    "price": float(row['Close']),
                    "volume": float(row['Volume'])
                }
                if kafka_producer:
                    kafka_producer.send("aapl_prices", value=event_data)
                    kafka_producer.flush()
                else:
                    print(f"[Producer SIM] (No Broker) Gửi: {event_data}")
                time.sleep(0.1)
            print("[Producer] Historical backfill via yfinance completed. Stopping producer thread.")
        except Exception as ex:
            print(f"[Producer] CRITICAL: yfinance backfill failed: {ex}")

    def run(self):
        # 1. Fetch keys from environment
        api_key = os.getenv("ALPACA_API_KEY", "YOUR_API_KEY_ID_HERE")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
        
        has_keys = (
            api_key and secret_key 
            and api_key != "YOUR_API_KEY_ID_HERE" 
            and secret_key != "YOUR_SECRET_KEY_HERE"
        )
        
        if has_keys:
            print("[Producer] Found Alpaca API Keys. Checking market status...")
            try:
                import alpaca_trade_api as tradeapi
                api = tradeapi.REST(
                    key_id=api_key,
                    secret_key=secret_key,
                    base_url="https://paper-api.alpaca.markets"
                )
                
                # Check clock
                clock = api.get_clock()
                
                print(f"[Producer] Connected to Alpaca REST API successfully.")
                print(f"[Producer] New York Time: {clock.timestamp}")
                print(f"[Producer] Market is open: {clock.is_open}")
                
                if clock.is_open:
                    print(f"[Producer] US Stock Market is OPEN. Streaming trades for {self.ticker} via WebSocket...")
                    from alpaca_trade_api.stream import Stream
                    self.alpaca_stream = Stream(
                        key_id=api_key,
                        secret_key=secret_key,
                        base_url="https://paper-api.alpaca.markets",
                        data_feed="iex"
                    )
                    
                    async def handle_trade(trade):
                        event_data = {
                            "ticker": self.ticker,
                            "timestamp": trade.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            "price": float(trade.price),
                            "volume": float(trade.size)
                        }
                        if kafka_producer:
                            kafka_producer.send("aapl_prices", value=event_data)
                            kafka_producer.flush()
                        else:
                            print(f"[Producer SIM] (No Broker) Gửi: {event_data}")
                    
                    self.alpaca_stream.subscribe_trades(handle_trade, self.ticker)
                    self.alpaca_stream.run()
                    return
                else:
                    print("[Producer] US Stock Market is CLOSED. Fetching hourly backfill from Alpaca...")
                    from alpaca_trade_api.rest import TimeFrame
                    from datetime import timedelta
                    end_dt = datetime.now()
                    start_dt = end_dt - timedelta(days=7)
                    
                    bars = api.get_bars(
                        self.ticker,
                        TimeFrame.Hour,
                        start=start_dt.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                        feed="iex"
                    ).df
                    
                    print(f"[Producer] Fetched {len(bars)} hourly bars from Alpaca. Sending to Kafka...")
                    
                    # Reset index to make 'timestamp' a column in DataFrame
                    bars = bars.reset_index()
                    for _, row in bars.iterrows():
                        event_data = {
                            "ticker": self.ticker,
                            "timestamp": row['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f"),
                            "price": float(row['close']),
                            "volume": float(row['volume'])
                        }
                        if kafka_producer:
                            kafka_producer.send("aapl_prices", value=event_data)
                            kafka_producer.flush()
                        else:
                            print(f"[Producer SIM] (No Broker) Gửi: {event_data}")
                        time.sleep(0.1)
                    print("[Producer] Historical backfill via Alpaca completed. Stopping producer thread.")
                    return
            except Exception as e:
                print(f"[Producer] Connection to Alpaca REST API failed: {e}")
                self.run_yfinance_backfill()
                return
        else:
            print("[Producer] Alpaca API Keys not configured or placeholders found in .env.")
            self.run_yfinance_backfill()
            return

    def stop(self):
        self.running = False
        if self.alpaca_stream:
            print("[Producer] Stopping Alpaca connection...")
            try:
                import asyncio
                # Gracefully stop Alpaca async stream
                asyncio.run(self.alpaca_stream.stop())
            except Exception:
                pass

# ----------------- SECTION 3: KAFKA CONSUMER & DATA LAKE SINK (HDFS Storage) -----------------
class DataLakeSinkConsumer(threading.Thread):
    def __init__(self, batch_size: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.buffer = []
        self.running = True
        self.consumer = None

    def run(self):
        print(f"[Consumer] Connecting to Kafka Broker at {KAFKA_BOOTSTRAP_SERVERS}...")
        try:
            self.consumer = KafkaConsumer(
                "aapl_prices",
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id="data-lake-sink-group",
                auto_offset_reset="earliest",
                enable_auto_commit=True
            )
            print("[Consumer] Connected to Kafka topic 'aapl_prices' successfully.")
        except Exception as e:
            print(f"[Consumer] CRITICAL: Failed to connect to Kafka: {e}")
            print("[Consumer] Please ensure Docker container is running.")
            return

        while self.running:
            # Poll messages from topic
            msg_pack = self.consumer.poll(timeout_ms=500)
            
            if msg_pack:
                for tp, messages in msg_pack.items():
                    for message in messages:
                        event_data = message.value
                        print(f"[Consumer] Polled event from Kafka: {event_data}")
                        self.buffer.append(event_data)
                        
                        # Flush to Data Lake JSON block
                        if len(self.buffer) >= self.batch_size:
                            self.write_to_data_lake()
            
            time.sleep(0.1)
            
        # Close consumer connection
        if self.consumer:
            self.consumer.close()
            
        # Flush remaining events upon shutdown
        if self.buffer:
            print(f"[Consumer] Flushing remaining {len(self.buffer)} events to Data Lake on exit...")
            self.write_to_data_lake()

    def write_to_data_lake(self):
        # Generate HDFS partition file name
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(DATA_LAKE_DIR, f"part_{timestamp_str}.json")
        
        print(f"\n[Data Lake Sink] >>> BUFFER FULL ({self.batch_size} events). FLUSHING TO DATA LAKE (HDFS): {file_path} <<<")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.buffer, f, ensure_ascii=False, indent=4)
            
        print("[Data Lake Sink] Flush completed successfully!\n")
        self.buffer = []

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    print("=== STARTING KAFKA STREAMING PIPELINE ===")
    
    producer = StockProducer("AAPL")
    consumer = DataLakeSinkConsumer(batch_size=5)
    
    producer.start()
    consumer.start()
    
    # Run the demo streaming for 15 seconds, then gracefully shut down
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        pass
    
    print("\n--- Shutting down threads... ---")
    producer.stop()
    consumer.running = False
    
    producer.join(timeout=2.0)
    consumer.join()
    
    print("=== PIPELINE SHUTDOWN COMPLETED ===")
