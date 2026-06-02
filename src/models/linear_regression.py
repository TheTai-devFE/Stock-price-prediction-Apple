import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.models.base_model import BaseModel

class LinearRegressionModel(BaseModel):
    """
    Wrapper for Scikit-Learn's Linear Regression model.
    """

    def __init__(self):
        super().__init__("LinearRegression")
        self.model = LinearRegression()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"[*] Training Linear Regression model...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        x_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(x_values)
