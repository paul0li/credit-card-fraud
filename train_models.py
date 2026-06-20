import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


TARGET = "is_fraud"
DEFAULT_TRAIN_PATH = "data/train_preprocessed.csv"
DEFAULT_TEST_PATH = "data/test_preprocessed.csv"
OUTPUT_DIR = "model_output"
RANDOM_STATE = 42


def load_xy(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"ROC AUC: {roc_auc_score(y_test, y_score):.4f}")
    print(f"PR AUC:  {average_precision_score(y_test, y_score):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_score, y_pred


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_score in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC comparativas")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitima", "Fraude"],
        yticklabels=["Legitima", "Fraude"],
        ax=ax,
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion - {name}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/confusion_matrix_{name}.png", dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train, y_train = load_xy(DEFAULT_TRAIN_PATH)
    X_test, y_test = load_xy(DEFAULT_TEST_PATH)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_depth=6,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
    }

    roc_results = {}
    for name, model in models.items():
        y_score, y_pred = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        roc_results[name] = y_score
        plot_confusion_matrix(y_test, y_pred, name)

    plot_roc_curves(roc_results, y_test)

if __name__ == "__main__":
    main()
