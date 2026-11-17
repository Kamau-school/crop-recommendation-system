"""
Part 5: Feature engineering.

Tests whether engineered features or removing weak/redundant features improve
cross-validated accuracy on the TRAINING set (the test set is never touched).

Run:  python -m src.features
Writes reports/feature_experiments.csv and models/feature_set.json (the chosen set).
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config

BASE = config.FEATURES
NEW = ["NPK_sum", "N_P_ratio", "N_K_ratio", "P_K_ratio", "temp_x_humidity", "rainfall_per_humidity"]

FEATURE_SETS = {
    "baseline (7 original)": BASE,
    "drop P (redundant with K)": [c for c in BASE if c != "P"],
    "drop K": [c for c in BASE if c != "K"],
    "drop ph and temperature (weakest)": ["N", "P", "K", "humidity", "rainfall"],
    "baseline + NPK_sum": BASE + ["NPK_sum"],
    "baseline + nutrient ratios": BASE + ["N_P_ratio", "N_K_ratio", "P_K_ratio"],
    "baseline + all 6 engineered": BASE + NEW,
}


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Create the new features (+1 in denominators avoids division by zero)."""
    X = X.copy()
    X["NPK_sum"] = X["N"] + X["P"] + X["K"]
    X["N_P_ratio"] = X["N"] / (X["P"] + 1)
    X["N_K_ratio"] = X["N"] / (X["K"] + 1)
    X["P_K_ratio"] = X["P"] / (X["K"] + 1)
    X["temp_x_humidity"] = X["temperature"] * X["humidity"] / 100
    X["rainfall_per_humidity"] = X["rainfall"] / (X["humidity"] + 1)
    return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Pipeline step: builds engineered features and keeps the chosen feature set.
    Being part of the saved pipeline, the web app applies identical logic to user input."""

    def __init__(self, feature_set="baseline (7 original)"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=BASE)
        return engineer(X)[FEATURE_SETS[self.feature_set]]


def load_train():
    df = pd.read_csv(config.PROCESSED_DIR / "train_unscaled.csv")
    return df[BASE], df["label"]


def main():
    X, y = load_train()
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1),
        "SVM": SVC(kernel="rbf", random_state=config.RANDOM_STATE),
    }
    rows = []
    for name, feats in FEATURE_SETS.items():
        row = {"feature_set": name, "n_features": len(feats)}
        for mname, clf in models.items():
            pipe = Pipeline([("features", FeatureSelector(name)), ("scaler", StandardScaler()), ("model", clf)])
            s = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            row[f"{mname} CV acc"], row[f"{mname} CV std"] = s.mean(), s.std()
        row["mean CV acc"] = np.mean([row["Random Forest CV acc"], row["SVM CV acc"]])
        rows.append(row)
        print(f"{name:38s} RF {row['Random Forest CV acc']:.4f} | SVM {row['SVM CV acc']:.4f}")
    res = pd.DataFrame(rows)

    # Decision rule: use a different set only if it beats the baseline by at least 0.2 percentage points
    base_acc = res.loc[0, "mean CV acc"]
    best = res.loc[res["mean CV acc"].idxmax()]
    chosen = best["feature_set"] if best["mean CV acc"] - base_acc >= 0.002 else res.loc[0, "feature_set"]
    res["chosen"] = res["feature_set"] == chosen
    config.REPORTS_DIR.mkdir(exist_ok=True)
    res.round(4).to_csv(config.REPORTS_DIR / "feature_experiments.csv", index=False)
    json.dump({"feature_set": chosen, "features": FEATURE_SETS[chosen]}, open(config.MODELS_DIR / "feature_set.json", "w"), indent=2)
    print(f"\nChosen feature set: {chosen}")


if __name__ == "__main__":
    main()
