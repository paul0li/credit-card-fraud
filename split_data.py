"""
Separa fraud_clean.csv en train/test respetando el orden temporal
(80/20, sin mezclar filas — conclusiones_eda.md, punto 7) y genera una
versión balanceada del train mediante submuestreo de la clase mayoritaria.
"""

import pandas as pd

INPUT_PATH = "data/fraud_clean.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TRAIN_BALANCED_PATH = "data/train_balanced.csv"

TRAIN_FRACTION = 0.8
RANDOM_STATE = 42


def temporal_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def balance_majority_class(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0].sample(n=len(fraud), random_state=random_state)
    return (
        pd.concat([fraud, legit])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    train, test = temporal_split(df)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"Train: {train.shape[0]:,} filas -> {TRAIN_PATH}")
    print(f"  is_fraud: {train['is_fraud'].value_counts().to_dict()}")
    print(f"Test:  {test.shape[0]:,} filas -> {TEST_PATH}")
    print(f"  is_fraud: {test['is_fraud'].value_counts().to_dict()}")

    train_balanced = balance_majority_class(train)
    train_balanced.to_csv(TRAIN_BALANCED_PATH, index=False)
    print(f"Train balanceado: {train_balanced.shape[0]:,} filas -> {TRAIN_BALANCED_PATH}")
    print(f"  is_fraud: {train_balanced['is_fraud'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
