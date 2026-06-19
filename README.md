**Problema**

Es importante que las empresas de tarjetas de crédito puedan reconocer transacciones fraudulentas, para que los clientes no sean cobrados por artículos que no compraron.


**Sobre el dataset**

Este es un dataset simulado de transacciones con tarjeta de crédito, con transacciones legítimas y fraudulentas entre el 1 de enero de 2019 y el 31 de diciembre de 2020. Cubre tarjetas de crédito de 1000 clientes haciendo transacciones con un conjunto de 800 comercios.

El dataset se encuentra en esta URL: [https://www.kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

**Configuración**

1. Instala [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Configura las credenciales de Kaggle (ver `.env.example`):
   - Ve a https://www.kaggle.com/settings/api → "Create New Token" → descarga `kaggle.json`.
   - Copia el archivo de ejemplo y completa con los valores de `kaggle.json`:
```bash
cp .env.example .env
# Edita .env con tu KAGGLE_USERNAME y KAGGLE_KEY de kaggle.json
```
3. Crea el entorno virtual:
```bash
uv sync
```
4. Ejecuta el EDA:
```bash
uv run analysis.py
```

## Preparación de datos

Pipeline en 4 pasos:

1. `uv run analysis.py` — EDA completo, gráficos y conclusiones en `eda_output/`.
2. `uv run clean_data.py` — selecciona columnas relevantes (`amt`, `trans_hour`, `age`, `category` en one-hot, `is_fraud`) y guarda `data/fraud_clean.csv`.
3. `uv run split_data.py` — divide en train/test y genera un train balanceado.
4. `uv run preprocess.py` — escala las features y guarda los conjuntos listos para modelar.

Los archivos generados (`data/`) no están en el repo — también se pueden descargar directo desde [Google Drive](https://drive.google.com/drive/folders/1a0cQr3LnwNbEhUXJPvkRwEby5c0SSyal?usp=drive_link).

### División train/test

La división es **temporal, no aleatoria** (las filas ya están en orden cronológico): primer 80% → train, último 20% → test. Un split aleatorio dejaría "ver el futuro" al modelo.

| Conjunto | Filas | Legítimas (0) | Fraude (1) | % fraude |
|---|---|---|---|---|
| `train.csv` | 1,037,340 | 1,031,372 | 5,968 | 0.575% |
| `test.csv` | 259,335 | 257,797 | 1,538 | 0.593% |
| `train_balanced.csv` | 11,936 | 5,968 | 5,968 | 50% |

- `train_balanced.csv`: submuestreo de la clase mayoritaria **solo en train** (1:1), para modelos sensibles al desbalance.
- `test.csv`: mantiene la distribución real (~0.6% fraude) — usar para métricas finales (PR-AUC, recall, precisión).

### Preprocesamiento (`preprocess.py`)

Toma `train_balanced.csv` y `test.csv` y aplica:

| Feature | Transformación | Razón |
|---|---|---|
| `amt` | log1p → StandardScaler | Distribución muy sesgada a la derecha (mediana $47 vs media $67 en legítimas); log1p comprime outliers antes de escalar |
| `trans_hour` | Codificación cíclica (sin/cos) | Variable circular: 23:59 debe estar "cerca" de 00:00; reemplaza la columna por `trans_hour_sin` y `trans_hour_cos` |
| `age` | StandardScaler | Distribución aproximadamente normal; escala a media 0 y desviación 1 |
| `category_*` | Sin cambios | Ya en one-hot binario desde `clean_data.py` |

El `StandardScaler` se ajusta **solo en train** y se aplica a test, evitando data leakage. El scaler se guarda en `data/scaler.joblib` para reutilizarlo en inferencia.

### Manejo del desbalance de clases

El dataset tiene ~0.58% de fraude — desbalance extremo. Se combinan dos estrategias complementarias:

1. **Undersampling en train** (`split_data.py`): submuestreo aleatorio de la clase mayoritaria hasta 1:1. Reduce el ruido de la clase dominante y acelera el entrenamiento. Se aplica solo a train; test conserva la distribución real.
2. **`class_weight` en el modelo** (recomendado para algunos algoritmos): penaliza más los errores en la clase minoritaria sin descartar datos. Compatible con Regresión Logística, Random Forest, y Gradient Boosting.

SMOTE se descartó por riesgo de data leakage: en datos con estructura temporal, generar muestras sintéticas interpolando entre vecinos puede filtrar información futura al pasado.

---

## Algoritmos investigados

Tabla comparativa teórica de los algoritmos candidatos para este problema de clasificación binaria con desbalance extremo y señal no lineal (ver `eda_output/conclusiones_eda.md`, punto 5).

| Algoritmo | Necesita escalado | `class_weight` | Interpretabilidad | Velocidad (12k filas) | Captura no linealidad | Evaluación para este problema |
|---|---|---|---|---|---|---|
| **Regresión Logística** | Sí | Sí | Alta (coeficientes) | Muy rápida | No | Buen baseline; falla si la señal de fraude es una combinación no lineal de features |
| **Random Forest** | No | Sí | Media (importancia de features) | Rápida | Sí | Robusto a outliers en `amt`; maneja bien el desbalance; candidato sólido |
| **Gradient Boosting** (XGBoost / LightGBM) | No | Sí (`scale_pos_weight`) | Media | Rápida (LightGBM especialmente) | Sí | Estado del arte en fraude con datos tabulares; **mejor candidato** según literatura |
| **SVM** (kernel RBF) | Sí | Sí | Baja | Lenta (O(n²)) | Sí | Costoso en memoria y tiempo incluso con 12k filas; poco práctico en producción |
| **Red Neuronal** (MLP) | Sí | Difícil de ajustar | Baja (caja negra) | Media | Sí | Overkill para 12k filas; podría valer la pena con el dataset completo (~1M filas) |

**Métricas de evaluación:** dado el desbalance extremo, la exactitud (*accuracy*) es engañosa. Usar **PR-AUC** (área bajo la curva Precisión-Recall) como métrica principal, complementada con recall y precisión a distintos umbrales de decisión.