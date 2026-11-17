"""
Part 7: Model evaluation on the held-out TEST set (never used for training or tuning).

Run:  python -m src.evaluate
Writes reports/model_comparison.csv, reports/classification_report.csv,
       reports/misclassifications.csv, reports/eval/*.png
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
OUT = config.REPORTS_DIR / "eval"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = joblib.load(config.MODELS_DIR / "all_models.pkl")
    le = joblib.load(config.LABEL_ENCODER_PATH)
    best = json.load(open(config.MODELS_DIR / "model_info.json"))["best_model"]
    cv = pd.read_csv(config.REPORTS_DIR / "model_cv_results.csv").set_index("model")

    test = pd.read_csv(config.PROCESSED_DIR / "test_unscaled.csv")
    X, y = test[config.FEATURES], test["label"]

    rows, preds = [], {}
    for name, pipe in models.items():
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X)
        preds[name] = pred
        rows.append({
            "Model": name,
            "CV accuracy": cv.loc[name, "cv_accuracy"],
            "Test accuracy": accuracy_score(y, pred),
            "Precision (macro)": precision_score(y, pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y, pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y, pred, average="macro", zero_division=0),
            "ROC-AUC (OvR)": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
            "Test errors": int((pred != y).sum()),
        })
    comp = pd.DataFrame(rows).sort_values(["Test accuracy", "F1 (macro)"], ascending=False)
    comp.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    print(comp.round(4).to_string(index=False))

    # Comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    m = comp.melt(id_vars="Model", value_vars=["CV accuracy", "Test accuracy", "F1 (macro)"])
    sns.barplot(data=m, y="Model", x="value", hue="variable", ax=ax, palette=["#b7d7c8", "#4f9d7e", "#1F4E3D"])
    ax.set_xlim(0.95, 1.001)
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    ax.set_title("Model comparison (axis starts at 0.95)")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    fig.savefig(OUT / "fig07_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix and per-class report for the selected model
    pred = preds[best]
    cm = confusion_matrix(y, pred)
    labels = le.classes_
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=np.where(cm == 0, "", cm), fmt="", cmap="Greens", xticklabels=labels,
                yticklabels=labels, cbar=False, linewidths=0.4, linecolor="#e5e5e5", ax=ax)
    ax.set_xlabel("Predicted crop")
    ax.set_ylabel("Actual crop")
    ax.set_title(f"Confusion matrix: {best} (test set, 440 records)")
    plt.xticks(rotation=90)
    fig.savefig(OUT / "fig08_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rep = pd.DataFrame(classification_report(y, pred, target_names=labels, output_dict=True, zero_division=0)).T
    rep.round(3).to_csv(config.REPORTS_DIR / "classification_report.csv")
    wrong = pd.DataFrame({"actual": le.inverse_transform(y[pred != y]),
                          "predicted": le.inverse_transform(pred[pred != y])})
    wrong.to_csv(config.REPORTS_DIR / "misclassifications.csv", index=False)
    print(f"\nSelected model: {best}. Misclassified {len(wrong)} of {len(y)} test records:")
    print(wrong.value_counts().to_string())

    # Does the choice of model matter? Which test records does any model get wrong?
    any_wrong = np.zeros(len(y), bool)
    for p in preds.values():
        any_wrong |= p != y
    print(f"Records misclassified by at least one model: {int(any_wrong.sum())}")


if __name__ == "__main__":
    main()
