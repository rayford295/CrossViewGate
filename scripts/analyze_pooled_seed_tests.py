from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pooled-across-seeds paired tests. Samples are the independent unit "
            "and seeds are repeated measures: for each sample we average the "
            "paired correctness difference over the seeds in which it qualifies "
            "(e.g. is in that seed's conflict subset), then test H0: E[delta]=0 "
            "with a sign-flip permutation test and a percentile bootstrap CI. "
            "This replaces 'significant in k/N seeds' reporting with one test."
        )
    )
    parser.add_argument("--multiseed-root", default="outputs/multiseed_main")
    parser.add_argument("--gate-predictions-dir", default="outputs/analysis/reliability_gate/predictions")
    parser.add_argument("--datasets", default="altadena_3class,ian_original,milton_original")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output-dir", default="outputs/analysis/pooled_seed_tests")
    parser.add_argument("--doc-path", default="docs/pooled_seed_tests.md")
    return parser.parse_args()


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def _numbered_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    pairs: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for column in df.columns:
        match = pattern.match(column)
        if match:
            pairs.append((int(match.group(1)), column))
    return [column for _, column in sorted(pairs)]


def sign_flip_p(deltas: np.ndarray, permutations: int, seed: int = 42) -> float:
    observed = abs(deltas.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        signs = rng.choice([-1.0, 1.0], size=len(deltas))
        if abs((deltas * signs).mean()) >= observed:
            count += 1
    return (count + 1) / (permutations + 1)


def bootstrap_ci(deltas: np.ndarray, rounds: int, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(rounds)
    for index in range(rounds):
        means[index] = deltas[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pooled_test(
    per_seed_deltas: list[pd.Series], permutations: int, bootstrap_rounds: int
) -> dict[str, object]:
    """per_seed_deltas: one Series per seed, indexed by sample_id, values in {-1,0,1}."""
    combined = pd.concat(per_seed_deltas, axis=1)
    per_sample = combined.mean(axis=1, skipna=True).dropna()
    deltas = per_sample.to_numpy(dtype=np.float64)
    if len(deltas) == 0:
        return {"n_samples": 0}
    ci_low, ci_high = bootstrap_ci(deltas, bootstrap_rounds)
    return {
        "n_samples": int(len(deltas)),
        "mean_delta": float(deltas.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_sign_flip": sign_flip_p(deltas, permutations),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.multiseed_root)
    gate_dir = Path(args.gate_predictions_dir)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        # method -> seed -> Series(sample_id -> correct)
        correct: dict[str, dict[int, pd.Series]] = {}
        conflict_masks: dict[int, pd.Series] = {}
        for seed in seeds:
            frames: dict[str, pd.DataFrame] = {}
            for mode in ("street_only", "remote_only", "concat", "crossview"):
                df = pd.read_csv(root / dataset / f"{mode}_seed{seed}" / "test_predictions.csv")
                df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
                frames[mode] = df.set_index("sample_id").sort_index()
                correct.setdefault(mode, {})[seed] = (
                    frames[mode]["prediction"] == frames[mode]["target"]
                ).astype(float)
            street_probs = frames["street_only"][_numbered_columns(frames["street_only"], "prob")].to_numpy()
            remote_probs = frames["remote_only"][_numbered_columns(frames["remote_only"], "prob")].to_numpy()
            prob_avg_pred = ((street_probs + remote_probs) / 2.0).argmax(axis=1)
            correct.setdefault("prob_average", {})[seed] = pd.Series(
                (prob_avg_pred == frames["street_only"]["target"].to_numpy()).astype(float),
                index=frames["street_only"].index,
            )
            conflict_masks[seed] = (
                frames["street_only"]["prediction"] != frames["remote_only"]["prediction"]
            )
            gate_path = gate_dir / f"{dataset}_seed{seed}.csv"
            if gate_path.exists():
                gate = pd.read_csv(gate_path)
                gate["sample_id"] = gate["sample_id"].map(_canonical_sample_id)
                gate = gate.set_index("sample_id").sort_index()
                for column, name in (
                    ("gate_linear_prediction", "gate_linear"),
                    ("gate3_linear_prediction", "gate3_linear"),
                ):
                    correct.setdefault(name, {})[seed] = (
                        gate[column] == gate["target"]
                    ).astype(float)

        comparisons = [
            ("crossview", "street_only", "conflict"),
            ("crossview", "remote_only", "conflict"),
            ("crossview", "concat", "conflict"),
            ("crossview", "prob_average", "conflict"),
            ("gate3_linear", "crossview", "conflict"),
            ("gate3_linear", "crossview", "full"),
            ("gate3_linear", "prob_average", "conflict"),
            ("gate_linear", "crossview", "conflict"),
            ("crossview", "street_only", "full"),
            ("crossview", "remote_only", "full"),
        ]
        for method_a, method_b, scope in comparisons:
            if method_a not in correct or method_b not in correct:
                continue
            per_seed: list[pd.Series] = []
            for seed in seeds:
                if seed not in correct[method_a] or seed not in correct[method_b]:
                    continue
                a = correct[method_a][seed]
                b = correct[method_b][seed]
                common = a.index.intersection(b.index)
                delta = a.loc[common] - b.loc[common]
                if scope == "conflict":
                    mask = conflict_masks[seed].reindex(common).fillna(False)
                    delta = delta[mask.to_numpy(dtype=bool)]
                per_seed.append(delta)
            result = pooled_test(per_seed, args.permutations, args.bootstrap)
            rows.append(
                {"dataset": dataset, "comparison": f"{method_a} - {method_b}", "scope": scope, **result}
            )
        print(f"tested {dataset}")

    results = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    results.to_csv(output_dir / "pooled_seed_tests.csv", index=False)

    lines = [
        "# Pooled-Across-Seeds Paired Tests",
        "",
        "One test per comparison instead of 'significant in k/N seeds'. Unit of",
        "analysis is the test sample; its paired correctness difference is averaged",
        "over the seeds in which it qualifies (conflict scope: seeds where the two",
        "single-view models disagree on it). H0: E[delta] = 0, two-sided sign-flip",
        "permutation test; percentile bootstrap 95% CI.",
        "",
        "| dataset | comparison | scope | n | mean delta | 95% CI | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in results.iterrows():
        if row.get("n_samples", 0) == 0:
            continue
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['scope']} | {row['n_samples']} | "
            f"{row['mean_delta']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['p_sign_flip']:.4f} |"
        )
    doc_path = Path(args.doc_path)
    ensure_dir(doc_path.parent)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {doc_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
