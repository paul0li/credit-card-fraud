"""
Limpieza del dataset de fraude: selecciona las columnas relevantes según
el EDA (eda_output/conclusiones_eda.md) y guarda un CSV listo para modelar.
"""

import os

import pandas as pd

from analysis import add_derived_columns, load_data

OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fraud_clean.csv")

# Columnas pedidas: monto, hora del día, edad y categoría de comercio
# (one-hot, 14 valores) — las 3 señales individuales más fuertes según el
# EDA (conclusiones_eda.md, "Conclusiones generales", punto 1), más la edad.
NUMERIC_COLUMNS = ["amt", "trans_hour", "age"]
CATEGORICAL_COLUMNS = ["category"]
TARGET_COLUMN = "is_fraud"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = add_derived_columns(df)

    dummies = pd.get_dummies(df[CATEGORICAL_COLUMNS], prefix=CATEGORICAL_COLUMNS)

    return pd.concat(
        [df[NUMERIC_COLUMNS], dummies, df[[TARGET_COLUMN]]],
        axis=1,
    )


def main() -> None:
    df = load_data()
    clean = clean_data(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset limpio guardado en {OUTPUT_PATH}")
    print(f"Forma: {clean.shape[0]:,} filas x {clean.shape[1]} columnas")
    print(clean.head())


if __name__ == "__main__":
    main()
