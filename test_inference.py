import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

imputer = joblib.load(BASE_DIR / "artifacts/imputer.pkl")
scaler = joblib.load(BASE_DIR / "artifacts/scaler.pkl")
selector = joblib.load(BASE_DIR / "artifacts/selector.pkl")
model = joblib.load(BASE_DIR / "artifacts/model.pkl")

sample = np.random.rand(21).reshape(1, -1)

sample = imputer.transform(sample)
sample = scaler.transform(sample)
sample = selector.transform(sample)

label_map = {
    1: "Normal",
    2: "Hyperthyroid",
    3: "Hypothyroid"
}

pred = int(model.predict(sample)[0])
print("Predicted Class:", label_map[pred])