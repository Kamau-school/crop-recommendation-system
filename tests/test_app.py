"""Smoke tests for the web app. Run from the project root:  python -m pytest -q  (or python tests/test_app.py)"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.app import app  # noqa: E402

client = app.test_client()
GOOD = {"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph": 6.5, "rainfall": 203}


def test_home_page():
    assert client.get("/").status_code == 200


def test_health():
    assert client.get("/health").get_json()["status"] == "ok"


def test_form_prediction_rice():
    r = client.post("/predict", data=GOOD)
    assert r.status_code == 200 and b"Rice" in r.data


def test_api_prediction():
    r = client.post("/api/predict", json=GOOD)
    assert r.status_code == 200
    assert r.get_json()["predictions"][0]["crop"] == "rice"


def test_invalid_input_rejected():
    bad = dict(GOOD, ph=20)
    assert client.post("/api/predict", json=bad).status_code == 400
    assert client.post("/api/predict", json={}).status_code == 400


def test_csv_upload_and_download():
    csv = (Path(__file__).resolve().parent.parent / "app" / "sample_input.csv").read_bytes()
    r = client.post("/predict_csv", data={"file": (io.BytesIO(csv), "x.csv")}, content_type="multipart/form-data")
    assert r.status_code == 200 and b"recommended_crop" in r.data
    assert client.get("/download").status_code == 200


def test_bad_csv_rejected():
    r = client.post("/predict_csv", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")}, content_type="multipart/form-data")
    assert r.status_code == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
