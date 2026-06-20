from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "duration_s",
    "packets",
    "bytes_total",
    "avg_packet_size",
    "interarrival_ms",
    "down_up_ratio",
    "throughput_kbps",
]


@dataclass
class ClassificationResult:
    accuracy: float
    macro_f1: float
    report: str
    confusion_matrix: list[list[int]]
    labels: list[str]


def train_classifier(data: pd.DataFrame, seed: int = 42) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = data[FEATURES]
    y = data["traffic_class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=220, max_depth=12, random_state=seed, n_jobs=-1)),
        ]
    )
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test


def evaluate_classifier(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> ClassificationResult:
    y_pred = model.predict(x_test)
    labels = sorted(y_test.unique().tolist())
    return ClassificationResult(
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        report=classification_report(y_test, y_pred),
        confusion_matrix=confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        labels=labels,
    )
