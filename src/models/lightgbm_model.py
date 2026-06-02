import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from src.models.base_model import BaseModel

class LightGBMModel(BaseModel):
    """
    Wrapper for LightGBM Regressor model.
    """

    def __init__(self, n_estimators=1000, learning_rate=0.05, max_depth=5, n_jobs=-1, random_state=42):
        super().__init__("LightGBM")
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, eval_set=None, early_stopping_rounds=None):
        print(f"[*] Training LightGBM model...")
        if eval_set is not None and early_stopping_rounds is not None:
            import lightgbm as lgb
            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=callbacks
            )
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        x_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(x_values)
