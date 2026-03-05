import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EngineerFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Nothing to learn, but required by sklearn API
        return self

    def transform(self, X):
        X = X.copy()

        # drop patient ID
        X = X.drop(columns=["patient_id"], errors="ignore")

        # age features
        X["is_elderly"] = (X["age"] >= 65).astype(int)
        X["age_group"] = pd.cut(
            X["age"],
            bins=[0, 40, 65, 80, 120],
            labels=["young", "adult", "senior", "elderly"]
        )

        # utilization intensity features
        X["procedures_per_day"] = X["procedures_count"] / (X["length_of_stay_days"] + 1)
        X["meds_per_day"] = X["medication_count"] / (X["length_of_stay_days"] + 1)

        # length of stay features
        X["log_LOS"] = np.log1p(X["length_of_stay_days"])
        X["long_stay"] = (X["length_of_stay_days"] > 7).astype(int)

        # cost feature (avoid divide by zero)
        X["cost_per_day"] = X["total_cost_€"] / (X["length_of_stay_days"] + 1)

        # satisfaction feature
        X["low_satisfaction"] = (X["satisfaction_score"] <= 3).astype(int)

        return X