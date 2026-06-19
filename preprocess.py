"""
Preprocesamiento para modelado:
- amt: log1p + StandardScaler (distribución fuertemente asimétrica a la derecha)
- trans_hour: codificación cíclica sin/cos (hora es variable circular: 23:00 cerca de 00:00)
- age: StandardScaler
- category_*: ya en one-hot desde clean_data.py, sin cambios

El escalador se ajusta únicamente en train para evitar data leakage.
Salidas: data/train_preprocessed.csv, data/test_preprocessed.csv, data/scaler.joblib
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "data/train_balanced.csv"
TEST_PATH = "data/test.csv"
TRAIN_OUT = "data/train_preprocessed.csv"
TEST_OUT = "data/test_preprocessed.csv"
SCALER_PATH = "data/scaler.joblib"

TARGET = "is_fraud"
SCALE_COLS = ["amt", "age"]


def cyclical_encode(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


def preprocess(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, StandardScaler]:
    X_train = train.drop(columns=[TARGET]).copy()
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET]).copy()
    y_test = test[TARGET]

    # amt: log1p comprime la distribución sesgada antes de escalar
    X_train["amt"] = np.log1p(X_train["amt"])
    X_test["amt"] = np.log1p(X_test["amt"])

    # trans_hour: codificación cíclica para que 23 esté cerca de 0
    X_train["trans_hour_sin"], X_train["trans_hour_cos"] = cyclical_encode(X_train["trans_hour"], 24)
    X_test["trans_hour_sin"], X_test["trans_hour_cos"] = cyclical_encode(X_test["trans_hour"], 24)
    X_train = X_train.drop(columns=["trans_hour"])
    X_test = X_test.drop(columns=["trans_hour"])

    # StandardScaler en amt y age — ajuste solo en train (evita data leakage)
    scaler = StandardScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])

    return X_train, y_train, X_test, y_test, scaler


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Train cargado: {train.shape} | is_fraud: {train[TARGET].value_counts().to_dict()}")
    print(f"Test cargado:  {test.shape}  | is_fraud: {test[TARGET].value_counts().to_dict()}")

    X_train, y_train, X_test, y_test, scaler = preprocess(train, test)

    os.makedirs("data", exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_OUT, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_OUT, index=False)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nColumnas resultantes ({len(X_train.columns)}): {list(X_train.columns)}")
    print(f"Train preprocesado: {X_train.shape} -> {TRAIN_OUT}")
    print(f"Test preprocesado:  {X_test.shape} -> {TEST_OUT}")
    print(f"Scaler guardado en: {SCALER_PATH}")


if __name__ == "__main__":
    main()
