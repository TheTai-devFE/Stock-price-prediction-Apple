import abc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BaseModel(abc.ABC):
    """
    Abstract Base Class for all machine learning models in the system.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abc.abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model.
        """
        pass

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Make predictions.
        """
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the model's performance on test data.
        Returns MSE, MAE, and R2 score.
        """
        try:
            predictions = self.predict(X_test, y=y_test)
        except TypeError:
            predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "predictions": predictions
        }
