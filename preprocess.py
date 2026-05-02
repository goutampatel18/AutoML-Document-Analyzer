"""
preprocess.py — Data Preprocessing Pipeline
AutoML Document Analyzer
Handles missing values, encoding, scaling, and train-test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────
# Problem-type detection
# ─────────────────────────────────────────────

def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    """
    Returns 'classification' if the target column is categorical or
    has ≤ 20 unique integer values; otherwise returns 'regression'.
    """
    series = df[target].dropna()

    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    n_unique = series.nunique()
    # Heuristic: few unique integers → treat as classification
    if n_unique <= 20 and pd.api.types.is_integer_dtype(series):
        return "classification"

    return "regression"


# ─────────────────────────────────────────────
# Preprocessing pipeline
# ─────────────────────────────────────────────

class Preprocessor:
    """
    End-to-end preprocessing pipeline.

    Steps:
      1. Drop columns with > 80 % missing values
      2. Impute remaining missing values (numeric → median, cat → most_frequent)
      3. Drop duplicates
      4. Encode the target column (classification only)
      5. One-hot encode low-cardinality categoricals (≤ 20 unique)
         Label-encode high-cardinality categoricals
      6. Standard-scale numeric features
      7. Train-test split (80/20, stratified for classification)
    """

    def __init__(self, problem_type: str = "classification",
                 test_size: float = 0.2, random_state: int = 42):
        self.problem_type = problem_type
        self.test_size = test_size
        self.random_state = random_state

        # Fitted artefacts (populated in .fit_transform())
        self.target_encoder: LabelEncoder | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.ohe_columns: list[str] = []
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.dropped_cols: list[str] = []
        self.report: dict = {}           # human-readable preprocessing log

    # ──────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame, target: str):
        """
        Fit the pipeline on *df* and return (X_train, X_test, y_train, y_test).
        Also populates self.report with a preprocessing summary.
        """
        df = df.copy()

        # ── 1. Drop high-missing columns (excl. target) ──────────────────────
        thresh = 0.80
        n = len(df)
        hi_miss = [c for c in df.columns
                   if c != target and df[c].isnull().mean() > thresh]
        df.drop(columns=hi_miss, inplace=True)
        self.dropped_cols = hi_miss

        # ── 2. Drop duplicates ────────────────────────────────────────────────
        n_before = len(df)
        df.drop_duplicates(inplace=True)
        n_dropped_dupes = n_before - len(df)

        # ── 3. Separate features and target ──────────────────────────────────
        X = df.drop(columns=[target])
        y = df[target].copy()

        # ── 4. Encode target (classification) ────────────────────────────────
        if self.problem_type == "classification":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target)
            self.target_encoder = le

        # ── 5. Identify column types ─────────────────────────────────────────
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # ── 6. Impute ────────────────────────────────────────────────────────
        if num_cols:
            num_imp = SimpleImputer(strategy="median")
            X[num_cols] = num_imp.fit_transform(X[num_cols])

        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            X[cat_cols] = cat_imp.fit_transform(X[cat_cols])

        # ── 7. Encode categorical features ───────────────────────────────────
        low_card = [c for c in cat_cols if X[c].nunique() <= 20]
        high_card = [c for c in cat_cols if X[c].nunique() > 20]
        self.ohe_columns = low_card

        # One-hot encode low-cardinality
        if low_card:
            X = pd.get_dummies(X, columns=low_card, drop_first=False)

        # Label-encode high-cardinality
        for col in high_card:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # ── 8. Scale numeric features ────────────────────────────────────────
        # Re-identify numeric cols after OHE (bool → int first)
        bool_cols = X.select_dtypes(include="bool").columns.tolist()
        X[bool_cols] = X[bool_cols].astype(int)

        num_cols_final = X.select_dtypes(include=np.number).columns.tolist()
        scaler = StandardScaler()
        X[num_cols_final] = scaler.fit_transform(X[num_cols_final])
        self.scaler = scaler

        self.feature_names = X.columns.tolist()

        # ── 9. Train-test split ───────────────────────────────────────────────
        stratify = y if self.problem_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            # Fallback: no stratify (happens with tiny datasets)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        # ── 10. Build report ─────────────────────────────────────────────────
        self.report = {
            "dropped_high_missing_cols": hi_miss,
            "duplicates_removed": n_dropped_dupes,
            "original_cat_cols": cat_cols,
            "low_card_ohe": low_card,
            "high_card_label_enc": high_card,
            "numeric_cols_scaled": num_cols_final,
            "final_feature_count": len(self.feature_names),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "target_classes": (
                self.target_encoder.classes_.tolist()
                if self.target_encoder else None
            ),
        }

        return X_train, X_test, y_train, y_test
