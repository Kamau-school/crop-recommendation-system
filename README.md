# Crop Recommendation System Using Machine Learning

AI Mini Project (group project). A supervised multi-class classifier that recommends one of
22 crops from seven soil and climate measurements (N, P, K, temperature, humidity, pH,
rainfall), delivered through a Flask web app.

**Group members:** Shaddy Jelimo (INTE/M/0063/01/25), Precious Ogolla (INTE/M/0374/01/25), Eric Kamau (INTE/M/0417/01/25)

## Results (held-out test set, 440 records)
| Model | CV accuracy | Test accuracy | F1 (macro) |
|---|---|---|---|
| **Random Forest (deployed)** | 0.9955 | 0.9955 | 0.9955 |
| Naive Bayes | 0.9949 | 0.9955 | 0.9954 |
| Gradient Boosting | 0.9920 | 0.9886 | 0.9887 |
| SVM (RBF) | 0.9830 | 0.9886 | 0.9886 |
| Logistic Regression | 0.9778 | 0.9841 | 0.9840 |
| KNN | 0.9682 | 0.9818 | 0.9816 |
| Decision Tree | 0.9858 | 0.9795 | 0.9794 |

The dataset appears to be generated within fixed ranges per crop, so these scores are likely higher than real-farm performance. See the report for limitations.

## Dataset
Kaggle: Crop Recommendation Dataset (Atharva Ingle)
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Download it and place `Crop_recommendation.csv` in `data/raw/`.

## Project structure
```
app/               Flask web app (app.py, templates/, static/, sample_input.csv)
data/raw/          original CSV (not committed)
data/processed/    cleaned data and train/test splits (created by preprocess.py)
models/            trained model, label encoder, scaler, settings (used by the web app)
notebooks/         Part 3, Part 4 and Parts 5-8 notebooks with outputs
reports/           figures and tables used in the report
src/               config, preprocess, eda, features, train, evaluate, interpret
tests/             smoke tests for the web app
Procfile           start command for Render / Heroku
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the whole pipeline
```bash
python -m src.preprocess         # Part 3: clean, encode, split, scale
python -m src.eda                # Part 4: EDA figures -> reports/eda/
python -m src.features           # Part 5: feature experiments -> models/feature_set.json
python -m src.train              # Part 6: tune and train 7 models -> models/
python -m src.evaluate           # Part 7: test-set metrics, confusion matrix -> reports/
python -m src.interpret          # Part 8: feature importance, crop ranges
```
The trained model is already included in `models/`, so you can skip straight to the app.

## Run the web app (Part 9)
```bash
python -m app.app                # open http://127.0.0.1:5000
python tests/test_app.py         # optional smoke tests
```
Features: enter values in a form, or upload a CSV (see `app/sample_input.csv`) and download the predictions.
JSON API: `POST /api/predict` with `{"N":90,"P":42,"K":43,"temperature":21,"humidity":82,"ph":6.5,"rainfall":203}`.

## Deploy on Render (free tier)
1. Push this folder to GitHub.
2. On render.com choose New > Web Service and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app.app:app`
5. Add the public URL to the report and slides.

## Links
- GitHub repository: _add link_
- Live app: _add link_
