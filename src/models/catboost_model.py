import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from src.models.base_model import BaseModel

class CatBoostModel(BaseModel):
    """
    Wrapper for CatBoost Regressor model.
    """

    def __init__(self, iterations=1000, learning_rate=0.05, depth=5, random_seed=42):
        super().__init__("CatBoost")
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=0
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, eval_set=None, early_stopping_rounds=None):
        print(f"[*] Training CatBoost model...")
        if eval_set is not None and early_stopping_rounds is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=0
            )
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        x_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(x_values)
