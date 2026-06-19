"""
EDA del dataset de Kaggle "Credit Card Fraud Detection" (Kartik Shenoy).
Variable objetivo (clasificación binaria): is_fraud (1 = fraude, 0 = legítimo).

Genera:
  1. Reporte de "descripción del dataset" en consola (forma, tipos, nulos,
     duplicados, cardinalidad, desbalance de clases).
  2. Conjunto de gráficos de EDA guardados como PNG en eda_output/.
  3. Tabla resumen final impresa en consola.
"""

import os

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = "eda_output"

# Descripción breve de lo que probablemente representa cada columna. Se usa en
# el reporte de "Descripción del dataset". Este dataset no tiene diccionario
# de datos oficial, así que estas descripciones se infieren de nombres/valores.
COLUMN_DESCRIPTIONS = {
    "Unnamed: 0": "Índice de fila heredado del CSV original (sin valor analítico)",
    "trans_date_trans_time": "Marca de tiempo de la transacción (fecha + hora)",
    "cc_num": "Número de tarjeta anonimizado (identifica al cliente/tarjeta)",
    "merchant": "Nombre del comercio donde ocurrió la transacción",
    "category": "Categoría del comercio / tipo de compra (ej. grocery_pos, misc_net)",
    "amt": "Monto de la transacción en dólares",
    "first": "Nombre del titular de la tarjeta",
    "last": "Apellido del titular de la tarjeta",
    "gender": "Género del titular de la tarjeta",
    "street": "Dirección (calle) del titular",
    "city": "Ciudad del titular",
    "state": "Estado del titular",
    "zip": "Código postal del titular",
    "lat": "Latitud del domicilio del titular",
    "long": "Longitud del domicilio del titular",
    "city_pop": "Población de la ciudad del titular",
    "job": "Ocupación del titular",
    "dob": "Fecha de nacimiento del titular",
    "trans_num": "Identificador único de la transacción",
    "unix_time": "Marca de tiempo de la transacción en segundos Unix (epoch)",
    "merch_lat": "Latitud del comercio al momento de la transacción",
    "merch_long": "Longitud del comercio al momento de la transacción",
    "is_fraud": "Etiqueta objetivo: 1 = transacción fraudulenta, 0 = legítima",
}


def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    csv_path = os.path.join(path, "fraudTrain.csv")
    return pd.read_csv(csv_path, encoding="latin-1")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas que necesitamos para el EDA: trans_hour y edad del cliente."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour

    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (
        (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    ).astype(int)
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    return df


# ---------------------------------------------------------------------------
# 1. Descripción del dataset
# ---------------------------------------------------------------------------

def describe_dataset(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DESCRIPCIÓN DEL DATASET")
    print("=" * 70)

    print(f"\nForma: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    print("\nTipos de dato:")
    print(df.dtypes)

    print("\nMuestra de filas:")
    print(df.head(5))

    print("\nDescripción de columnas (inferida):")
    for col in df.columns:
        desc = COLUMN_DESCRIPTIONS.get(col, "(sin descripción disponible)")
        print(f"  - {col}: {desc}")

    print("\nValores nulos por columna:")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(nulls if not nulls.empty else "  Ninguno — el dataset no tiene nulos")

    n_dupes = df.duplicated().sum()
    print(f"\nFilas duplicadas: {n_dupes:,}")

    print("\nCardinalidad de columnas categóricas:")
    cat_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in cat_cols:
        print(f"  - {col}: {df[col].nunique():,} valores únicos")

    print("\nDesbalance de clases (is_fraud):")
    counts = df["is_fraud"].value_counts().sort_index()
    pct = df["is_fraud"].value_counts(normalize=True).sort_index() * 100
    for label, count in counts.items():
        name = "Fraude" if label == 1 else "Legítima"
        print(f"  - {name} ({label}): {count:,} ({pct[label]:.3f}%)")


def plot_class_imbalance(df: pd.DataFrame) -> None:
    counts = df["is_fraud"].value_counts().sort_index()
    labels = ["Legítima (0)", "Fraude (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=labels, y=counts.values, hue=labels, palette=["#4C72B0", "#C44E52"], legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Número de transacciones (escala log)")
    ax.set_title("Desbalance de clases: fraude vs. transacciones legítimas")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    save_fig(fig, "01_class_imbalance.png")
    # Hallazgo: el fraude es una minoría diminuta (<1%) — desbalance de clases
    # extremo clásico, así que la exactitud (accuracy) por sí sola será una
    # métrica engañosa para cualquier modelo posterior.


# ---------------------------------------------------------------------------
# 2. EDA — distribuciones
# ---------------------------------------------------------------------------

def plot_amount_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name, color in [(0, "Legítima", "#4C72B0"), (1, "Fraude", "#C44E52")]:
        subset = df.loc[df["is_fraud"] == label, "amt"]
        sns.histplot(subset, bins=60, log_scale=True, stat="density",
                     color=color, label=name, alpha=0.5, ax=ax)
    ax.set_xlabel("Monto de la transacción ($, escala log)")
    ax.set_title("Distribución del monto: fraude vs. legítima")
    ax.legend()
    save_fig(fig, "02_amount_distribution.png")
    # Hallazgo: los montos de fraude se concentran en valores más altos que las
    # transacciones legítimas típicas — las compras fraudulentas tienden a ser
    # más grandes en promedio.


def plot_hour_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x="trans_hour", hue="is_fraud", bins=24, multiple="dodge",
                 stat="proportion", common_norm=False, palette=["#4C72B0", "#C44E52"], ax=ax)
    ax.set_xlabel("Hora del día")
    ax.set_title("Distribución por hora de la transacción: fraude vs. legítima (normalizado por clase)")
    ax.set_xticks(range(0, 24))
    save_fig(fig, "03_hour_distribution.png")
    # Hallazgo: el fraude se concentra desproporcionadamente en horas de
    # madrugada/noche, mientras que las transacciones legítimas siguen un
    # patrón típico de compras diurnas.


def plot_top_categories_by_fraud_rate(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    rates = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)
    top = rates.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top.values * 100, y=top.index, hue=top.index, palette="rocket", legend=False, ax=ax)
    ax.set_xlabel("Tasa de fraude (%)")
    ax.set_ylabel("Categoría de comercio")
    ax.set_title(f"Top {top_n} categorías de comercio por tasa de fraude")
    save_fig(fig, "04_top_categories_by_fraud_rate.png")
    # Hallazgo: categorías como compras online / misc-net tienden a mostrar las
    # tasas de fraude más altas — los canales sin presencia física de la
    # tarjeta son más riesgosos.
    return rates


# ---------------------------------------------------------------------------
# 2. EDA — correlaciones y relaciones
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long",
                    "unix_time", "trans_hour", "age", "is_fraud"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Mapa de correlación de variables numéricas")
    save_fig(fig, "05_correlation_heatmap.png")
    # Hallazgo: la mayoría de las variables numéricas tienen correlación débil
    # con is_fraud individualmente (amt muestra la correlación positiva más
    # fuerte, aunque modesta) — la señal de fraude depende más de
    # combinaciones/patrones que de variables lineales aisladas.


def plot_amount_boxplot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="is_fraud", y="amt", hue="is_fraud",
                palette=["#4C72B0", "#C44E52"], legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legítima", "Fraude"])
    ax.set_ylabel("Monto de la transacción ($, escala log)")
    ax.set_title("Monto de la transacción por clase")
    save_fig(fig, "06_amount_boxplot.png")
    # Hallazgo: las transacciones de fraude muestran una mediana de monto más
    # alta y un rango intercuartílico (IQR) más estrecho y desplazado hacia
    # arriba que las legítimas.


def plot_fraud_rate_by_category(df: pd.DataFrame) -> None:
    rates = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False) * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=rates.values, y=rates.index, hue=rates.index, palette="mako", legend=False, ax=ax)
    ax.set_xlabel("Tasa de fraude (%)")
    ax.set_ylabel("Categoría de comercio")
    ax.set_title("Tasa de fraude por categoría de comercio (ordenado)")
    save_fig(fig, "07_fraud_rate_by_category.png")
    # Hallazgo: la tasa de fraude varía considerablemente entre categorías —
    # un puñado de categorías concentra un riesgo de fraude desproporcionado
    # respecto a su volumen.


def note_geographic_data(df: pd.DataFrame) -> None:
    has_geo = {"lat", "long", "merch_lat", "merch_long"}.issubset(df.columns)
    if has_geo:
        print("\nNota: hay campos geográficos presentes (lat, long, merch_lat, merch_long).")
        print("Podrían soportar features basadas en distancia (ej. distancia entre")
        print("cliente y comercio), pero no se mapean aquí — fuera del alcance de esta tarea.")


# ---------------------------------------------------------------------------
# 2. EDA — estadísticas clave
# ---------------------------------------------------------------------------

def report_amount_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("is_fraud")["amt"].agg(["mean", "median", "std"])
    stats.index = ["Legítima", "Fraude"]
    print("\nEstadísticas del monto de transacción por clase:")
    print(stats)
    return stats


def plot_fraud_rate_by_hour(df: pd.DataFrame) -> None:
    rates = df.groupby("trans_hour")["is_fraud"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=rates.index, y=rates.values, marker="o", color="#C44E52", ax=ax)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Tasa de fraude (%)")
    ax.set_title("Tasa de fraude por hora del día")
    ax.set_xticks(range(0, 24))
    save_fig(fig, "08_fraud_rate_by_hour.png")
    # Hallazgo: la tasa de fraude tiene un pico marcado en horas de madrugada —
    # señal temporal fuerte y accionable para un modelo o un filtro basado en reglas.


def plot_fraud_rate_by_age_bucket(df: pd.DataFrame) -> None:
    rates = df.groupby("age_bucket", observed=True)["is_fraud"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=rates.index.astype(str), y=rates.values, hue=rates.index.astype(str),
                palette="flare", legend=False, ax=ax)
    ax.set_xlabel("Grupo de edad del cliente")
    ax.set_ylabel("Tasa de fraude (%)")
    ax.set_title("Tasa de fraude por grupo de edad del cliente")
    save_fig(fig, "09_fraud_rate_by_age_bucket.png")
    # Hallazgo: la tasa de fraude no es uniforme entre grupos de edad — los
    # segmentos de clientes de mayor edad tienden a mostrar tasas algo más
    # altas, lo que sugiere que la edad podría ser una feature útil.


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame) -> None:
    total = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_pct = fraud_count / total * 100
    mean_fraud_amt = df.loc[df["is_fraud"] == 1, "amt"].mean()
    mean_legit_amt = df.loc[df["is_fraud"] == 0, "amt"].mean()

    print("\n" + "=" * 70)
    print("TABLA RESUMEN")
    print("=" * 70)
    summary = pd.DataFrame({
        "Métrica": ["Transacciones totales", "Cantidad de fraudes", "% de fraude",
                    "Monto promedio fraude ($)", "Monto promedio legítimo ($)"],
        "Valor": [f"{total:,}", f"{fraud_count:,}", f"{fraud_pct:.3f}%",
                  f"{mean_fraud_amt:.2f}", f"{mean_legit_amt:.2f}"],
    })
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def save_fig(fig, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado: {path}")


def main() -> None:
    sns.set_theme(style="whitegrid")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    df = add_derived_columns(df)

    # 1. Descripción del dataset
    describe_dataset(df)
    plot_class_imbalance(df)

    # 2. EDA — distribuciones
    plot_amount_distribution(df)
    plot_hour_distribution(df)
    plot_top_categories_by_fraud_rate(df)

    # 2. EDA — correlaciones y relaciones
    plot_correlation_heatmap(df)
    plot_amount_boxplot(df)
    plot_fraud_rate_by_category(df)
    note_geographic_data(df)

    # 2. EDA — estadísticas clave
    report_amount_stats(df)
    plot_fraud_rate_by_hour(df)
    plot_fraud_rate_by_age_bucket(df)

    # Resumen final
    print_summary_table(df)


if __name__ == "__main__":
    main()
