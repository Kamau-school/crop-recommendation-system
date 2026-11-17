"""
Part 6: Model building.

Trains several classifiers, tuning each with 5-fold cross-validated Grid Search on the
TRAINING set only. Every model is a full pipeline (feature set -> scaler -> classifier),
so scaling is learned inside each CV fold (no leakage) and the web app can reuse it as-is.

Run:  python -m src.train        (after preprocess and features)
Writes models/all_models.pkl, models/crop_model.pkl (best by CV accuracy),
       reports/model_cv_results.csv
"""
import json
import time
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.features import FeatureSelector, load_train

warnings.filterwarnings("ignore", category=FutureWarning)   # SVC(probability=True) deprecation notice
SEED = config.RANDOM_STATE

CANDIDATES = {
    "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=SEED),
                            {"C": [1, 10, 100]}),
    "KNN": (KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}),
    "SVM (RBF)": (SVC(kernel="rbf", probability=True, random_state=SEED),
                  {"C": [1, 10, 100], "gamma": ["scale", 0.1]}),
    "Naive Bayes": (GaussianNB(),
                    {"var_smoothing": [1e-9, 1e-7, 1e-5]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                      {"n_estimators": [200], "max_depth": [None, 20], "min_samples_split": [2, 5]}),
    "Gradient Boosting": (HistGradientBoostingClassifier(random_state=SEED),
                          {"learning_rate": [0.05, 0.1], "max_iter": [100]}),
}


def main():
    X, y = load_train()
    feature_set = json.load(open(config.MODELS_DIR / "feature_set.json"))["feature_set"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    fitted, rows = {}, []
    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("features", FeatureSelector(feature_set)),
                         ("scaler", StandardScaler()), ("model", clf)])
        search = GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()},
                              cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
        t0 = time.time()
        search.fit(X, y)
        secs = time.time() - t0
        best_i = search.best_index_
        fitted[name] = search.best_estimator_
        rows.append({"model": name,
                     "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                     "cv_accuracy": search.best_score_,
                     "cv_std": search.cv_results_["std_test_score"][best_i],
                     "search_seconds": round(secs, 1)})
        print(f"{name:20s} CV acc {search.best_score_:.4f} (+/- {rows[-1]['cv_std']:.4f})  "
              f"{search.best_params_}  [{secs:.0f}s]", flush=True)

    res = pd.DataFrame(rows).sort_values(["cv_accuracy", "cv_std"], ascending=[False, True])
    config.REPORTS_DIR.mkdir(exist_ok=True)
    res.round(4).to_csv(config.REPORTS_DIR / "model_cv_results.csv", index=False)

    best_name = res.iloc[0]["model"]            # chosen by CV accuracy, not by the test set
    joblib.dump(fitted, config.MODELS_DIR / "all_models.pkl", compress=3)
    joblib.dump(fitted[best_name], config.MODELS_DIR / "crop_model.pkl", compress=3)
    json.dump({"best_model": best_name, "feature_set": feature_set},
              open(config.MODELS_DIR / "model_info.json", "w"), indent=2)
    print(f"\nBest model by cross-validation: {best_name}")


if __name__ == "__main__":
    main()
