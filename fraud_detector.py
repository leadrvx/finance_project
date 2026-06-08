from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "is_fraud"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_ESTIMATORS = 200
MIN_HIST_BINS = 10
MAX_HIST_BINS = 25


@dataclass
class ModelArtifacts:
    model: RandomForestClassifier
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    metrics: dict[str, float | None]


def generate_sample_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    amount = rng.exponential(scale=120, size=n_samples).round(2)
    hour = rng.integers(0, 24, size=n_samples)
    merchant_category = rng.choice(["grocery", "electronics", "travel", "fashion"], size=n_samples)
    international = rng.choice([0, 1], p=[0.9, 0.1], size=n_samples)
    card_present = rng.choice([0, 1], p=[0.25, 0.75], size=n_samples)
    prior_fraud_count = rng.poisson(0.25, size=n_samples)

    risk_score = (
        (amount > 300).astype(float) * 1.0
        + (hour < 6).astype(float) * 0.5
        + international * 1.4
        + (1 - card_present) * 0.9
        + prior_fraud_count * 0.8
        + (merchant_category == "travel").astype(float) * 0.6
        + rng.normal(0, 0.25, size=n_samples)
    )

    prob = 1.0 / (1.0 + np.exp(-(risk_score - 2.2)))
    is_fraud = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "merchant_category": merchant_category,
            "international": international,
            "card_present": card_present,
            "prior_fraud_count": prior_fraud_count,
            TARGET_COLUMN: is_fraud,
        }
    )


def run_eda(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing")

    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().sort_values(ascending=False),
        "class_distribution": df[target_column].value_counts(dropna=False),
        "numeric_summary": df.describe(include=[np.number]),
    }


def create_visualizations(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[plt.Figure]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing")

    figures: list[plt.Figure] = []

    fig_dist, ax_dist = plt.subplots(figsize=(5, 4))
    counts = df[target_column].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax_dist, color=["#4c78a8", "#f58518"])
    ax_dist.set_title("Répartition fraude / non-fraude")
    ax_dist.set_xlabel(target_column)
    ax_dist.set_ylabel("Nombre de transactions")
    figures.append(fig_dist)

    if "amount" in df.columns:
        fig_amount, ax_amount = plt.subplots(figsize=(6, 4))
        base_bins = int(df.shape[0] ** 0.5)
        if df.shape[0] < MIN_HIST_BINS:
            bins = max(2, base_bins)
        else:
            bins = min(MAX_HIST_BINS, max(MIN_HIST_BINS, base_bins))
        for label in sorted(df[target_column].dropna().unique()):
            subset = df.loc[df[target_column] == label, "amount"]
            ax_amount.hist(subset, bins=bins, alpha=0.6, label=f"{target_column}={label}")
        ax_amount.set_title("Distribution des montants par classe")
        ax_amount.set_xlabel("Montant")
        ax_amount.set_ylabel("Fréquence")
        ax_amount.legend()
        figures.append(fig_amount)

    return figures


def prepare_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing")

    y = df[target_column].astype(int)
    X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
    X = X.fillna(0)
    return X, y


def train_model(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    random_state: int = 42,
    test_size: float = DEFAULT_TEST_SIZE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> ModelArtifacts:
    X, y = prepare_features(df, target_column=target_column)

    if y.nunique() < 2:
        raise ValueError("Model training needs at least 2 classes in the target")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    y_proba = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X_test))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
        if y_test.nunique() > 1 and proba.shape[1] > 1
        else None,
    }

    return ModelArtifacts(model=model, X_test=X_test, y_test=y_test, y_pred=y_pred, metrics=metrics)


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="black")

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig
