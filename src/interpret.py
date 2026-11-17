"""
Part 8: Results interpretation.

Run:  python -m src.interpret
Writes reports/feature_importance.csv, reports/crop_requirements.csv, reports/eval/fig09_feature_importance.png
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def main():
    out = config.REPORTS_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.MODELS_DIR / "crop_model.pkl")
    best = json.load(open(config.MODELS_DIR / "model_info.json"))["best_model"]
    test = pd.read_csv(config.PROCESSED_DIR / "test_unscaled.csv")
    X, y = test[config.FEATURES], test["label"]

    # Permutation importance: how much does test accuracy drop when one feature is shuffled?
    perm = permutation_importance(model, X, y, n_repeats=30, random_state=config.RANDOM_STATE, scoring="accuracy")
    imp = pd.DataFrame({"feature": config.FEATURES, "accuracy_drop": perm.importances_mean,
                        "std": perm.importances_std})
    # Built-in importance (impurity-based) when the model has it
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        imp["impurity_importance"] = est.feature_importances_
    imp = imp.sort_values("accuracy_drop", ascending=False)
    imp.round(4).to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    print(imp.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp["feature"][::-1], imp["accuracy_drop"][::-1], xerr=imp["std"][::-1], color="#4f9d7e")
    ax.set_xlabel("Drop in test accuracy when the feature is shuffled")
    ax.set_title(f"Permutation feature importance ({best})")
    fig.savefig(out / "fig09_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Agronomic summary: typical range per crop (min to max in the full dataset)
    df = pd.read_csv(config.PROCESSED_DIR / "crop_clean.csv")
    g = df.groupby(config.TARGET)[config.FEATURES]
    lo, hi = g.min(), g.max()
    req = pd.DataFrame({c: lo[c].round(1 if c in ("temperature", "humidity", "ph") else 0).astype(str)
                           + " to " + hi[c].round(1 if c in ("temperature", "humidity", "ph") else 0).astype(str)
                        for c in config.FEATURES})
    req.to_csv(config.REPORTS_DIR / "crop_requirements.csv")
    # Training ranges (used by the web app to warn about out-of-range inputs)
    json.dump({c: [float(df[c].min()), float(df[c].max())] for c in config.FEATURES},
              open(config.MODELS_DIR / "feature_ranges.json", "w"), indent=2)


if __name__ == "__main__":
    main()
