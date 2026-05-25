import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Class for Stage 5: Feature Engineering.
    Generates technical indicators and lag features.
    """

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators (SMA, EMA, RSI, MACD).
        """
        print("[*] Generating Technical Indicators...")
        data = df.copy()

        # 1. Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()

        # 2. RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # 3. MACD (Moving Average Convergence Divergence)
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

        return data

    @staticmethod
    def add_lag_features(df: pd.DataFrame, lags: list = [1, 5, 7]) -> pd.DataFrame:
        """
        Add lag features for the closing price.
        """
        print(f"[*] Generating Lag Features for lags: {lags}...")
        data = df.copy()
        for lag in lags:
            data[f'lag_{lag}'] = data['close'].shift(lag)
        
        # Drop rows with NaN values created by lagging/rolling
        data = data.dropna()
        return data

    def process_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        df = self.add_technical_indicators(df)
        df = self.add_lag_features(df)
        print(f"[*] Feature Engineering completed. New shape: {df.shape}")
        return df
