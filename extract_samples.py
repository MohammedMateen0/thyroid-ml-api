import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data/ann-train.data",
    sep=r"\s+",
    header=None
)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

for label in [1, 2, 3]:
    sample = X[y == label].iloc[0].tolist()
    print(f"\nClass {label} sample:")
    print(sample)