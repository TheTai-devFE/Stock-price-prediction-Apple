import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class EDAVisualizer:
    """
    Class for Stage 4: EDA (Exploratory Data Analysis).
    Handles plotting stock trends and correlations.
    """
    
    def __init__(self, output_dir: str = "reports/eda"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Style settings
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [12, 6]

    def plot_price_trend(self, df: pd.DataFrame, ticker: str):
        """Plot the stock price trend over time."""
        plt.figure()
        plt.plot(df['date'], df['close'], label='Close Price', color='#1f77b4')
        plt.title(f"{ticker} Stock Price Trend", fontsize=16, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        
        output_path = os.path.join(self.output_dir, "price_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Exported Price Trend to: {output_path}")

    def plot_volume_trend(self, df: pd.DataFrame, ticker: str):
        """Plot the trading volume over time."""
        plt.figure()
        plt.bar(df['date'], df['volume'], color='#2ca02c', alpha=0.7)
        plt.title(f"{ticker} Trading Volume Trend", fontsize=16, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Volume")
        
        output_path = os.path.join(self.output_dir, "volume_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Exported Volume Trend to: {output_path}")

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap between numeric features."""
        plt.figure(figsize=(10, 8))
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        
        sns.heatmap(corr, annot=True, cmap='RdBu_r', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
        
        output_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Exported Correlation Heatmap to: {output_path}")
