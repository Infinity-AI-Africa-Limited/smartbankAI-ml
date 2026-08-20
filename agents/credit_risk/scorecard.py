"""Importable serialisation contract for the SmartBank development credit scorecard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


NUMERIC_FEATURES = [
    "monthly_income_ngn", "loan_amount_requested_ngn", "loan_tenure_months",
    "existing_monthly_obligations_ngn", "repayment_history_score", "account_age_months",
    "avg_monthly_balance_ngn",
]
CATEGORICAL_FEATURES = ["employment_type", "bvn_verified"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class WoeScorecard:
    """Serialisable WOE transform plus calibrated logistic scorecard."""

    numeric_edges: dict[str, list[float]]
    mappings: dict[str, dict[str, float]]
    model: LogisticRegression

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed: dict[str, np.ndarray] = {}
        for feature in NUMERIC_FEATURES:
            edges = np.array(self.numeric_edges[feature])
            values = pd.to_numeric(frame[feature], errors="coerce").fillna(0).to_numpy()
            bins = np.digitize(values, edges[1:-1], right=True)
            mapping = self.mappings[feature]
            transformed[feature] = np.array([mapping.get(str(index), 0.0) for index in bins])
        for feature in CATEGORICAL_FEATURES:
            mapping = self.mappings[feature]
            transformed[feature] = frame[feature].astype(str).map(mapping).fillna(0.0).to_numpy()
        return pd.DataFrame(transformed, index=frame.index)[FEATURES]

    def probability_of_default(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.transform(frame))[:, 1]
