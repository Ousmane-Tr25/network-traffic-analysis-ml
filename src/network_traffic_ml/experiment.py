from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data import generate_flow_dataset
from .model import FEATURES, evaluate_classifier, train_classifier


def run_experiment(output_dir: str | Path = "results", n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = generate_flow_dataset(n_samples=n_samples, seed=seed)
    model, _, x_test, _, y_test = train_classifier(data, seed=seed)
    result = evaluate_classifier(model, x_test, y_test)
    classifier = model.named_steps["classifier"]

    data.to_csv(output_path / "synthetic_network_flows.csv", index=False)
    metrics = pd.DataFrame([{"accuracy": result.accuracy, "macro_f1": result.macro_f1}])
    metrics.to_csv(output_path / "classification_metrics.csv", index=False)
    (output_path / "classification_report.txt").write_text(result.report, encoding="utf-8")
    pd.DataFrame(result.confusion_matrix, index=result.labels, columns=result.labels).to_csv(output_path / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(result.confusion_matrix)
    ax.set_title("Traffic classification confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(result.labels)))
    ax.set_yticks(range(len(result.labels)))
    ax.set_xticklabels(result.labels, rotation=25)
    ax.set_yticklabels(result.labels)
    for i, row in enumerate(result.confusion_matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    importance = pd.DataFrame({"feature": FEATURES, "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(output_path / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(importance["feature"], importance["importance"])
    ax.set_title("Network-flow feature importance")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path / "feature_importance.png", dpi=300)
    plt.close(fig)

    return metrics
