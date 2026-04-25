"""Lightweight evaluation harness for SMILES->pKd prediction.

Usage:
    python backend/scripts/evaluate_binding_affinity_hf.py --csv data.csv --smiles-col smiles --target-col pkd
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ligand_binding_predictor import LigandBindingPredictor

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    MurckoScaffold = None
    RDKIT_AVAILABLE = False


@dataclass
class EvalConfig:
    csv_path: str
    smiles_col: str
    target_col: str
    test_fraction: float
    random_state: int
    max_rows: int


def scaffold_for_smiles(smiles: str) -> str:
    if not RDKIT_AVAILABLE:
        return smiles[:24]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "INVALID"
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "NO_SCAFFOLD"
    except Exception:
        return "INVALID"


def bemis_murcko_split(
    smiles_list: Sequence[str],
    test_fraction: float,
    random_state: int,
) -> Tuple[List[int], List[int]]:
    scaffold_to_indices: Dict[str, List[int]] = {}
    for idx, smiles in enumerate(smiles_list):
        scaffold = scaffold_for_smiles(smiles)
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    groups = list(scaffold_to_indices.values())
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)

    total = len(smiles_list)
    target_test_size = max(1, int(total * test_fraction))
    test_indices: List[int] = []

    for group in groups:
        if len(test_indices) >= target_test_size:
            break
        test_indices.extend(group)

    test_index_set = set(test_indices)
    train_indices = [i for i in range(total) if i not in test_index_set]
    test_indices = sorted(test_index_set)
    return train_indices, test_indices


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    pearson_r = float(y_true_series.corr(y_pred_series, method="pearson"))
    spearman_rho = float(y_true_series.corr(y_pred_series, method="spearman"))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }


def evaluate(config: EvalConfig) -> Dict[str, float]:
    df = pd.read_csv(config.csv_path)
    if config.max_rows > 0:
        df = df.head(config.max_rows)

    df = df.dropna(subset=[config.smiles_col, config.target_col]).copy()
    df[config.smiles_col] = df[config.smiles_col].astype(str)
    df[config.target_col] = pd.to_numeric(df[config.target_col], errors="coerce")
    df = df.dropna(subset=[config.target_col])

    if len(df) < 2:
        raise ValueError("Need at least 2 valid rows for evaluation")

    smiles_values = df[config.smiles_col].tolist()
    train_idx, test_idx = bemis_murcko_split(
        smiles_values,
        test_fraction=config.test_fraction,
        random_state=config.random_state,
    )

    if not test_idx:
        raise ValueError("Scaffold split produced an empty test set")

    test_df = df.iloc[test_idx].reset_index(drop=True)

    predictor = LigandBindingPredictor()
    preds = predictor.predict_batch(test_df[config.smiles_col].tolist(), test_df[config.smiles_col].tolist())

    y_true: List[float] = []
    y_pred: List[float] = []
    for row, pred in zip(test_df.to_dict(orient="records"), preds):
        if not pred.get("is_valid"):
            continue
        y_true.append(float(row[config.target_col]))
        y_pred.append(float(pred.get("prediction", {}).get("binding_affinity", 0.0)))

    if len(y_true) < 2:
        raise ValueError("Not enough valid predictions to compute metrics")

    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    metrics.update(
        {
            "train_size": float(len(train_idx)),
            "test_size": float(len(test_idx)),
            "evaluated_size": float(len(y_true)),
        }
    )
    return metrics


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate HF binding-affinity predictor with scaffold split")
    parser.add_argument("--csv", required=True, help="Path to CSV with SMILES and pKd columns")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name")
    parser.add_argument("--target-col", default="pkd", help="Target pKd column name")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for quick runs")
    args = parser.parse_args()

    return EvalConfig(
        csv_path=args.csv,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    cfg = parse_args()
    results = evaluate(cfg)
    print("Binding Affinity Evaluation (Scaffold Split)")
    for key, value in results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
