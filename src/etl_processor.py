import pandas as pd
import numpy as np

class ETLProcessor:
    """
    Class for Stage 3: ETL (Extract, Transform, Load).
    Handles data cleaning and preparation.
    Designed for scalability.
    """

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning: handling missing values and outliers.
        """
        print("[*] Starting ETL process...")
        
        # 1. Clone to avoid modifying original
        processed_df = df.copy()

        # 2. Check for missing values
        null_counts = processed_df.isnull().sum().sum()
        if null_counts > 0:
            print(f"[*] Found {null_counts} missing values. Filling with Forward Fill...")
            processed_df = processed_df.fillna(method='ffill')
        
        # 3. Drop duplicates
        initial_count = len(processed_df)
        processed_df = processed_df.drop_duplicates()
        if len(processed_df) < initial_count:
            print(f"[*] Dropped {initial_count - len(processed_df)} duplicate rows.")

        # 4. Ensure column names are standardized (lowercase)
        processed_df.columns = [str(col).lower().replace(' ', '_').strip() for col in processed_df.columns]

        # 5. Handle potential outliers (Simple approach: Z-score or Clipping)
        # Here we just ensure numeric columns are correct
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_cols:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Drop any remaining rows with NaN after coercion
        processed_df = processed_df.dropna()

        print("[*] ETL process completed.")
        return processed_df
