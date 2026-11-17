"""
Part 3: Data cleaning and preprocessing.

Run from the project root:
    python -m src.preprocess

Reads   data/raw/Crop_recommendation.csv
Writes  data/processed/  (cleaned data and train/test splits)
        models/label_encoder.pkl and models/scaler.pkl (reused by the web app)
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config


def load_data(path=config.RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle 'Crop Recommendation Dataset' "
            "and place Crop_recommendation.csv in data/raw/."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates and data types; run validity checks."""
    df = df.copy()

    # Data types: features numeric, label tidy text
    for col in config.FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[config.TARGET] = df[config.TARGET].astype(str).str.strip().str.lower()

    # Missing values: drop rows with no label, fill numeric gaps with the median
    n_missing = int(df.isnull().sum().sum())
    if n_missing:
        df = df.dropna(subset=[config.TARGET])
        df[config.FEATURES] = df[config.FEATURES].fillna(df[config.FEATURES].median())

    # Duplicates
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Validity checks (physically possible values)
    invalid = (
        ~df["ph"].between(0, 14)
        | ~df["humidity"].between(0, 100)
        | (df[["N", "P", "K"]] < 0).any(axis=1)
        | (df["rainfall"] < 0)
    )
    print(f"Missing values found: {n_missing} | Duplicates removed: {n_dup} | "
          f"Invalid rows: {int(invalid.sum())}")
    return df[~invalid].reset_index(drop=True)


def iqr_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count IQR outliers per feature, overall and within each crop.
    Outliers are reported, not removed: they are genuine crop-specific values."""
    rows = []
    for col in config.FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        overall = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        within = 0
        for _, g in df.groupby(config.TARGET):
            gq1, gq3 = g[col].quantile([0.25, 0.75])
            giqr = gq3 - gq1
            within += int(((g[col] < gq1 - 1.5 * giqr) | (g[col] > gq3 + 1.5 * giqr)).sum())
        rows.append({"feature": col, "outliers_overall": overall, "outliers_within_crop": within})
    return pd.DataFrame(rows)


def build_splits(df: pd.DataFrame):
    """Encode the label, split 80/20 (stratified), scale using the training set only."""
    X = df[config.FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df[config.TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)          # fit on train only: no data leakage
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=config.FEATURES, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=config.FEATURES, index=X_test.index)
    return le, scaler, X_train, X_test, X_train_s, X_test_s, y_train, y_test


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = clean(load_data())
    print("\nOutlier report (kept, not removed):")
    print(iqr_outlier_report(df).to_string(index=False))

    le, scaler, X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te = build_splits(df)

    df.to_csv(config.PROCESSED_DIR / "crop_clean.csv", index=False)
    X_tr.assign(label=y_tr).to_csv(config.PROCESSED_DIR / "train_unscaled.csv", index=False)
    X_te.assign(label=y_te).to_csv(config.PROCESSED_DIR / "test_unscaled.csv", index=False)
    X_tr_s.assign(label=y_tr).to_csv(config.PROCESSED_DIR / "train_scaled.csv", index=False)
    X_te_s.assign(label=y_te).to_csv(config.PROCESSED_DIR / "test_scaled.csv", index=False)
    joblib.dump(le, config.LABEL_ENCODER_PATH)
    joblib.dump(scaler, config.SCALER_PATH)

    print(f"\nClasses: {len(le.classes_)} | Train: {X_tr.shape} | Test: {X_te.shape}")
    print(f"Saved processed data to {config.PROCESSED_DIR}")
    print(f"Saved label encoder and scaler to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
