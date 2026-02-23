import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_data(train_path, test_path):

    train_path = BASE_DIR / train_path
    test_path = BASE_DIR / test_path

    train = pd.read_csv(train_path, sep=r"\s+", header=None)
    test = pd.read_csv(test_path, sep=r"\s+", header=None)

    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    print("Raw df shape:", df.shape)
    print(df.head())

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("Shape:", X.shape)
    

    return X, y