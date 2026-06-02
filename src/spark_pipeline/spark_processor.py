from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import pandas as pd
import glob
import os
from datetime import datetime

class SparkProcessor:
    """
    Class for Stage 3 & 5 (Real-time Pipeline):
    ETL & Feature Engineering using Apache Spark.
    Reads partition files from the Data Lake and prepares them for models.
    """
    def __init__(self):
        # Initialize a local Spark Session
        self.spark = SparkSession.builder \
            .appName("AppleStockSparkProcessor") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        # Suppress verbose Spark logging
        self.spark.sparkContext.setLogLevel("ERROR")

    def initialize_data_lake(self, mongo_uri: str = "mongodb://localhost:27017/") -> bool:
        """
        Export historical raw data from MongoDB 'aapl_raw' collection
        to 'data/data_lake/' to bootstrap the Data Lake with historical data.
        This ensures Spark window features (like sma_50) calculate correctly.
        """
        from pymongo import MongoClient
        import json
        
        try:
            client = MongoClient(mongo_uri)
            db = client["apple_stock_db"]
            collection = db["aapl_raw"]
            
            cursor = collection.find({}, {"_id": 0})
            records = list(cursor)
            
            if not records:
                print("[SparkProcessor] MongoDB raw collection is empty. Cannot bootstrap.")
                return False
                
            os.makedirs("data/data_lake", exist_ok=True)
            chunk_size = 500
            
            # Write historical chunks as json files simulating real HDFS blocks
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i+chunk_size]
                formatted_chunk = []
                for r in chunk:
                    # Parse timestamp format
                    dt = r.get("Date") or r.get("date")
                    if isinstance(dt, datetime):
                        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                    else:
                        dt_str = str(dt)
                        
                    # Extract close price and volume
                    price = r.get("close") or r.get("Close")
                    volume = r.get("volume") or r.get("Volume") or 0.0
                    
                    formatted_chunk.append({
                        "ticker": "AAPL",
                        "timestamp": dt_str,
                        "price": float(price) if price else 0.0,
                        "volume": float(volume)
                    })
                    
                file_path = f"data/data_lake/part_historical_{i//chunk_size}.json"
                # Write to data lake
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(formatted_chunk, f, indent=4)
                    
            print(f"[SparkProcessor] Bootstrapped Data Lake with {len(records)} records across {len(records)//chunk_size + 1} files.")
            return True
        except Exception as e:
            print(f"[SparkProcessor] Error bootstrapping Data Lake: {e}")
            return False

    def process_data_lake(self, data_lake_dir: str = "data/data_lake") -> pd.DataFrame:
        """
        Read JSON partitions from the Data Lake, run Spark SQL analytics,
        and generate features identical to MongoDB aapl_features.
        """
        json_pattern = os.path.join(data_lake_dir, "part_*.json")
        if not glob.glob(json_pattern):
            print(f"[SparkProcessor] No partition files found in {data_lake_dir}")
            return pd.DataFrame()

        print(f"[SparkProcessor] Reading partition files from: {json_pattern}")
        
        # Clear catalog cache to force Spark to discover new partition files on disk
        self.spark.catalog.clearCache()
        
        # 1. Load JSON files into Spark DataFrame
        df = self.spark.read.option("multiLine", "true").json(json_pattern)
        
        # 2. Schema standardization & cleaning (ETL)
        df = df.withColumn("timestamp", F.to_timestamp("timestamp")) \
               .withColumn("price", F.col("price").cast("double")) \
               .withColumn("volume", F.col("volume").cast("double"))
               
        # Deduplicate records by timestamp to prevent duplicate events in data lake
        df = df.dropDuplicates(["timestamp"])
        
        # Sort values by time sequence
        df = df.orderBy("timestamp")
        
        # 3. Create Windows for time-series analytics
        window_all = Window.orderBy("timestamp")
        
        # 4. Map 'price' to standard OHLC/Close columns to match training schema
        df = df.withColumnRenamed("price", "close")
        df = df.withColumn("open", F.col("close"))
        df = df.withColumn("high", F.col("close"))
        df = df.withColumn("low", F.col("close"))
        df = df.withColumn("adj_close", F.col("close"))
        
        # 5. Technical Indicators
        # Moving Averages (SMA & EMA)
        df = df.withColumn("sma_20", F.avg("close").over(window_all.rowsBetween(-19, 0)))
        df = df.withColumn("sma_50", F.avg("close").over(window_all.rowsBetween(-49, 0)))
        df = df.withColumn("ema_20", F.avg("close").over(window_all.rowsBetween(-19, 0))) # Simple EMA approximation
        
        # Lags
        df = df.withColumn("lag_1", F.lag("close", 1).over(window_all))
        df = df.withColumn("lag_5", F.lag("close", 5).over(window_all))
        df = df.withColumn("lag_7", F.lag("close", 7).over(window_all))
        
        # RSI (Relative Strength Index)
        df = df.withColumn("prev_close", F.lag("close", 1).over(window_all))
        df = df.withColumn("diff", F.col("close") - F.col("prev_close"))
        
        df = df.withColumn("gain", F.when(F.col("diff") > 0, F.col("diff")).otherwise(0.0))
        df = df.withColumn("loss", F.when(F.col("diff") < 0, -F.col("diff")).otherwise(0.0))
        
        df = df.withColumn("avg_gain", F.avg("gain").over(window_all.rowsBetween(-13, 0)))
        df = df.withColumn("avg_loss", F.avg("loss").over(window_all.rowsBetween(-13, 0)))
        
        df = df.withColumn("rs", F.col("avg_gain") / F.col("avg_loss"))
        df = df.withColumn("rsi", F.when(F.col("avg_loss") == 0, 100.0)
                                   .otherwise(100.0 - (100.0 / (1.0 + F.col("rs")))))
        
        # MACD (Moving Average Convergence Divergence)
        df = df.withColumn("ema_12", F.avg("close").over(window_all.rowsBetween(-11, 0)))
        df = df.withColumn("ema_26", F.avg("close").over(window_all.rowsBetween(-25, 0)))
        df = df.withColumn("macd", F.col("ema_12") - F.col("ema_26"))
        df = df.withColumn("macd_signal", F.avg("macd").over(window_all.rowsBetween(-8, 0)))
        
        # Remove intermediate calculations and unused columns (like adj_close and ticker)
        df = df.drop("prev_close", "diff", "gain", "loss", "avg_gain", "avg_loss", "rs", "ema_12", "ema_26", "ticker", "adj_close")
        
        # Remove null values produced by lag & rolling operations
        df = df.dropna()
        
        # Convert Column timestamp to date to match schema of ML models
        df = df.withColumnRenamed("timestamp", "date")
        
        # Convert Spark DataFrame to Pandas DataFrame
        pandas_df = df.toPandas()
        
        print(f"[SparkProcessor] Completed processing. DataFrame shape: {pandas_df.shape}")
        return pandas_df
