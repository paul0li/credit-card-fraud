**Problem**

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.


**About the Dataset**

This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

The dataset can be found in this URL: [https://www.kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

**Setup**

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Configura las credenciales de Kaggle (ver `.env.example`):
   - Ve a https://www.kaggle.com/settings/api → "Create New Token" → descarga `kaggle.json`.
   - Copia el archivo de ejemplo y completa con los valores de `kaggle.json`:
```bash
cp .env.example .env
# Edita .env con tu KAGGLE_USERNAME y KAGGLE_KEY de kaggle.json
```
3. Create the virtual environment:
```bash
uv sync
```
4. Run the EDA:
```bash
uv run analysis.py
```

## Preparación de datos

Pipeline en 3 pasos:

1. `uv run analysis.py` — EDA completo, gráficos y conclusiones en `eda_output/`.
2. `uv run clean_data.py` — selecciona columnas relevantes (`amt`, `trans_hour`, `age`, `category` en one-hot, `is_fraud`) y guarda `data/fraud_clean.csv`.
3. `uv run split_data.py` — divide en train/test y genera un train balanceado.

### División train/test

La división es **temporal, no aleatoria** (las filas ya están en orden cronológico): primer 80% → train, último 20% → test. Un split aleatorio dejaría "ver el futuro" al modelo.

| Conjunto | Filas | Legítimas (0) | Fraude (1) | % fraude |
|---|---|---|---|---|
| `train.csv` | 1,037,340 | 1,031,372 | 5,968 | 0.575% |
| `test.csv` | 259,335 | 257,797 | 1,538 | 0.593% |
| `train_balanced.csv` | 11,936 | 5,968 | 5,968 | 50% |

- `train_balanced.csv`: submuestreo de la clase mayoritaria **solo en train** (1:1), para modelos sensibles al desbalance.
- `test.csv`: mantiene la distribución real (~0.6% fraude) — usar para métricas finales (PR-AUC, recall, precisión).