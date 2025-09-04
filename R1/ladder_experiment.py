from __future__ import annotations

import os
import json
from typing import List, Dict, Any

import numpy as np
import tqdm

from metrics import Metrics

ALGORITHMS = ["ISOMAP", "LLE", "PCA", "MDS", "UMAP", "TSNE", "RANDOM"]
RUN_IDX = 0
K = 46  # number of perturbation levels
MAX_SHUFFLE_FRAC = 0.90
RESULTS_JSON = os.path.join("R1", "ladder_results.json")
RNG_SEED = 1337

# Log-uniform scale range
LOG_ALPHA_LOW = -1.0
LOG_ALPHA_HIGH = 1.0


def _existing_datasets(dataset_dir: str) -> List[str]:
    """
    Return base names of all datasets.
    """
    names: List[str] = []
    if not os.path.isdir(dataset_dir):
        return names
    for fname in os.listdir(dataset_dir):
        if fname.endswith(".npy"):
            names.append(os.path.splitext(fname)[0])
    return sorted(names)


def _shuffle_subset_rows(
    Y: np.ndarray, frac: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Permute a fraction of rows in Y.
    """
    n = Y.shape[0]
    m = int(np.floor(frac * n))
    if m < 2:
        return Y.copy()

    idx = rng.choice(n, size=m, replace=False)
    perm = rng.permutation(idx)
    Yp = Y.copy()
    Yp[idx] = Y[perm]
    return Yp


def _sample_shared_alpha(rng: np.random.Generator) -> float:
    """
    Sample a single random scale.
    """
    return float(10.0 ** rng.uniform(LOG_ALPHA_LOW, LOG_ALPHA_HIGH))


def run_experiment() -> Dict[str, Any]:
    os.makedirs("R1", exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    datasets_dir = "datasets"
    embeddings_dir = "embeddings"

    datasets = _existing_datasets(datasets_dir)

    shuffle_fracs = np.linspace(0.0, MAX_SHUFFLE_FRAC, K)

    rows: List[Dict[str, Any]] = []

    total = sum(len(ALGORITHMS) * K for _ in datasets)
    with tqdm.tqdm(total=total) as pbar:
        for dataset in datasets:
            X = np.load(os.path.join(datasets_dir, f"{dataset}.npy"))

            # Pre-load base embeddings
            base_embeddings: Dict[str, np.ndarray] = {}
            for alg in ALGORITHMS:
                emb_path = os.path.join(
                    embeddings_dir, f"{dataset}_{alg}_{RUN_IDX}.npy"
                )
                base_embeddings[alg] = np.load(emb_path)

            # Perturbation ladder
            for sfrac in shuffle_fracs:
                alpha = _sample_shared_alpha(rng)

                for alg in ALGORITHMS:
                    Y = base_embeddings[alg]

                    Yp = _shuffle_subset_rows(Y, sfrac, rng)

                    # Apply random scale
                    Yps = alpha * Yp

                    # Compute metrics
                    M = Metrics(X, Yps, setbatch=False, precomputed=False)
                    NS = float(M.compute_normalized_stress())
                    SNS = float(M.compute_scale_normalized_stress())

                    rows.append(
                        {
                            "dataset": dataset,
                            "alg": alg,
                            "shuffle_frac": float(sfrac),
                            "alpha": float(alpha),
                            "NS": NS,
                            "SNS": SNS,
                        }
                    )

                    pbar.update(1)

    # Build output
    out: Dict[str, Any] = {
        "meta": {
            "algorithms": ALGORITHMS,
            "shuffle_fracs": [float(x) for x in shuffle_fracs],
            "datasets": datasets,
        },
        "results": rows,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


if __name__ == "__main__":
    run_experiment()
