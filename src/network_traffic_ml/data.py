from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
CLASSES = ("iot", "video", "interactive", "web")


def _sample_positive_normal(rng: np.random.Generator, mean: float, std: float, size: int) -> np.ndarray:
    return np.maximum(rng.normal(mean, std, size), 0.001)


def generate_flow_dataset(n_samples: int = 3000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic network-flow records for traffic classification."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(CLASSES, size=n_samples, p=[0.28, 0.30, 0.22, 0.20])
    rows = []
    for label in labels:
        if label == "iot":
            duration = _sample_positive_normal(rng, 1.8, 0.8, 1)[0]
            packets = rng.poisson(8) + 1
            bytes_total = _sample_positive_normal(rng, 900, 300, 1)[0]
            interarrival = _sample_positive_normal(rng, 180, 50, 1)[0]
            down_up_ratio = _sample_positive_normal(rng, 1.2, 0.4, 1)[0]
        elif label == "video":
            duration = _sample_positive_normal(rng, 45, 14, 1)[0]
            packets = rng.poisson(900) + 50
            bytes_total = _sample_positive_normal(rng, 5_000_000, 1_000_000, 1)[0]
            interarrival = _sample_positive_normal(rng, 18, 6, 1)[0]
            down_up_ratio = _sample_positive_normal(rng, 6.0, 1.4, 1)[0]
        elif label == "interactive":
            duration = _sample_positive_normal(rng, 16, 6, 1)[0]
            packets = rng.poisson(260) + 20
            bytes_total = _sample_positive_normal(rng, 700_000, 160_000, 1)[0]
            interarrival = _sample_positive_normal(rng, 8, 3, 1)[0]
            down_up_ratio = _sample_positive_normal(rng, 1.7, 0.5, 1)[0]
        else:
            duration = _sample_positive_normal(rng, 8, 3, 1)[0]
            packets = rng.poisson(120) + 10
            bytes_total = _sample_positive_normal(rng, 250_000, 90_000, 1)[0]
            interarrival = _sample_positive_normal(rng, 35, 12, 1)[0]
            down_up_ratio = _sample_positive_normal(rng, 3.0, 1.0, 1)[0]

        rows.append(
            {
                "duration_s": duration,
                "packets": packets,
                "bytes_total": bytes_total,
                "avg_packet_size": bytes_total / packets,
                "interarrival_ms": interarrival,
                "down_up_ratio": down_up_ratio,
                "throughput_kbps": (bytes_total * 8 / max(duration, 0.001)) / 1000,
                "traffic_class": label,
            }
        )
    return pd.DataFrame(rows).round(4)
