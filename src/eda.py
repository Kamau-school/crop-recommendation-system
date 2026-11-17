"""
Part 4: Exploratory Data Analysis.

Run from the project root (after python -m src.preprocess):
    python -m src.eda

Reads   data/processed/crop_clean.csv
Writes  reports/eda/*.png and *.csv  (figures and tables used in the report)
"""
import matplotlib
matplotlib.use("Agg")                      # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif

from src import config

FEATURES, TARGET = config.FEATURES, config.TARGET
OUT = config.REPORTS_DIR / "eda"
sns.set_theme(style="whitegrid")


def _save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load():
    path = config.PROCESSED_DIR / "crop_clean.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.preprocess` first.")
    return pd.read_csv(path)


def class_distribution(df):
    counts = df[TARGET].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    counts.plot.barh(ax=ax, color="#7fb69e")
    ax.set_xlabel("Number of records")
    ax.set_title("Records per crop")
    _save(fig, "fig01_class_distribution.png")


def feature_histograms(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), FEATURES):
        sns.histplot(df[c], bins=30, kde=True, ax=ax, color="#4f9d7e")
        ax.set_title(f"{c} (skew {df[c].skew():.2f})")
        ax.set_xlabel("")
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    _save(fig, "fig02_feature_histograms.png")


def correlation_heatmap(df):
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation between features")
    _save(fig, "fig03_correlation_heatmap.png")


def crop_profile_heatmap(df):
    means = df.groupby(TARGET)[FEATURES].mean()
    z = (means - means.mean()) / means.std()            # compare features on one scale
    labels = means.round(0).astype(int).astype(str)
    labels["ph"] = means["ph"].round(1).astype(str)       # pH needs a decimal place
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(z, annot=labels, fmt="", cmap="RdBu_r", center=0,
                cbar_kws={"label": "Standardised mean (colour)"}, ax=ax)
    ax.set_title("Average conditions for each crop (numbers = raw means)")
    ax.set_ylabel("")
    _save(fig, "fig04_crop_profile_heatmap.png")


def boxplots_by_crop(df):
    order = df.groupby(TARGET)["rainfall"].median().sort_values().index
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, c in zip(axes.ravel(), ["rainfall", "humidity", "K", "ph"]):
        sns.boxplot(data=df, x=TARGET, y=c, order=order, ax=ax, color="#7fb69e", fliersize=2)
        ax.set_title(f"{c} by crop")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    _save(fig, "fig05_boxplots_by_crop.png")


def scatter_relationships(df):
    pairs = [("P", "K"), ("humidity", "rainfall"), ("temperature", "humidity")]
    palette = sns.color_palette("husl", df[TARGET].nunique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (x, y) in zip(axes, pairs):
        sns.scatterplot(data=df, x=x, y=y, hue=TARGET, palette=palette, s=18, legend=False, ax=ax)
        ax.set_title(f"{y} vs {x}")
    handles = [plt.Line2D([], [], marker="o", ls="", color=col) for col in palette]
    fig.legend(handles, sorted(df[TARGET].unique()), loc="lower center", ncol=11, frameon=False, fontsize=9)
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, "fig06_scatter_relationships.png")


def summary_tables(df):
    stats = df[FEATURES].describe().T
    stats["skew"] = df[FEATURES].skew()
    stats.round(2).to_csv(OUT / "summary_statistics.csv")
    f, p = f_classif(df[FEATURES], df[TARGET])
    anova = pd.DataFrame({"feature": FEATURES, "F_statistic": f.round(0), "p_value": p}).sort_values(
        "F_statistic", ascending=False)
    anova.to_csv(OUT / "anova_f_scores.csv", index=False)
    print(stats.round(2).to_string())
    print("\nANOVA F-scores (higher = separates crops better):")
    print(anova.to_string(index=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    class_distribution(df)
    feature_histograms(df)
    correlation_heatmap(df)
    crop_profile_heatmap(df)
    boxplots_by_crop(df)
    scatter_relationships(df)
    summary_tables(df)


if __name__ == "__main__":
    main()
