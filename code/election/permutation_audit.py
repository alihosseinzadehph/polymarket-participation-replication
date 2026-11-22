"""Reproduce and extend the election allocation permutation inference.

Two related analyses are kept distinct:

1. The previously reported within-tier tests permute Harris/Trump labels among
   October-active wallet vectors separately in Bottom and Top strata.  They
   test candidate-label separation within one tier.  The exploratory family is
   eight tests: richness plus three profile JSDs in each tier.
2. The direct interaction test uses

       Delta = JSD(Harris_Bottom, Trump_Bottom)
               - JSD(Harris_Top, Trump_Top)

   for each of three profiles and three complete edge representations.  Each
   randomization independently reassigns candidate labels within each tier,
   preserving the observed active group sizes.  The full wallet vectors remain
   fixed, so the null conditions on the tier-specific spend, breadth, subject
   allocation, and concentration distributions.  The specified directional
   alternative is Delta > 0; Holm adjustment covers the six views.  This is a
   retrospective audit, not a preregistered analysis.

The compact input has no wallet addresses.  Only NumPy and the Python standard
library are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PACKAGE / "data/Source_Data/Fig6_8_election/election_october_wallet_vectors.csv"
DEFAULT_METADATA = PACKAGE / "data/Source_Data/Fig6_8_election/election_october_wallet_vectors_metadata.json"
DEFAULT_OUTPUT = Path.cwd() / "reproduced/election_permutation_audit.json"
DEFAULT_TABLE = Path.cwd() / "reproduced/election_direct_permutation.tex"
DEFAULT_CSV = Path.cwd() / "reproduced/election_direct_permutation.csv"

LEGACY_SEED = 20260925
DIRECT_SEED = 20260929
PERMUTATIONS = 9999
TIERS = ("Bottom 5%", "Top 5%")
METRICS = (
    "amount_profile",
    "equal_wallet_profile",
    "prevalence_profile",
    "amount_edges",
    "equal_wallet_edges",
    "binary_edges",
)
DISPLAY = {
    "amount_profile": "Amount profile",
    "equal_wallet_profile": "Equal-wallet profile",
    "prevalence_profile": "Prevalence profile",
    "amount_edges": "Amount edges",
    "equal_wallet_edges": "Equal-wallet edges",
    "binary_edges": "Binary edges",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--table-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--direct-seed", type=int, default=DIRECT_SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def holm_adjust(pvalues: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum(axis=-1, keepdims=True)
    return np.divide(values, total, out=np.zeros_like(values), where=total > 0)


def jsd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize(left)
    right = normalize(right)
    midpoint = 0.5 * (left + right)
    answer = np.zeros(left.shape[:-1], dtype=float)
    for values in (left, right):
        term = np.zeros_like(values)
        mask = values > 0
        term[mask] = values[mask] * np.log2(values[mask] / midpoint[mask])
        answer += 0.5 * term.sum(axis=-1)
    return answer


def load_vectors(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        subjects = reader.fieldnames[4:]
        rows = list(reader)
    groups: dict[str, list[list[float]]] = {"HB": [], "TB": [], "HT": [], "TT": []}
    for row in rows:
        cohort = row["cohort"]
        if cohort not in groups:
            raise ValueError(f"Unexpected cohort {cohort!r}")
        groups[cohort].append([float(row[column]) for column in subjects])
    arrays = {key: np.asarray(value, dtype=float) for key, value in groups.items()}
    for key, matrix in arrays.items():
        if not len(matrix) or matrix.shape[1] != len(subjects):
            raise ValueError(f"Invalid matrix for {key}: {matrix.shape}")
        if np.any(matrix < 0) or np.any(matrix.sum(axis=1) <= 0):
            raise ValueError(f"{key} contains negative or October-inactive rows")
    return subjects, arrays


def profile_set(matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix[matrix.sum(axis=1) > 0]
    proportions = matrix / matrix.sum(axis=1, keepdims=True)
    prevalence = (matrix > 0).mean(axis=0)
    return {
        "amount": normalize(matrix.sum(axis=0)),
        "equal_wallet": proportions.mean(axis=0),
        "prevalence": normalize(prevalence),
    }


def permutation_jsd(
    left: np.ndarray, right: np.ndarray, replicates: int, rng: np.random.Generator
) -> dict[str, dict[str, float | int]]:
    left_profiles = profile_set(left)
    right_profiles = profile_set(right)
    observed = {
        key: float(jsd(left_profiles[key], right_profiles[key]))
        for key in left_profiles
    }
    combined = np.vstack([left, right])
    left_size = len(left)
    exceed = {key: 0 for key in observed}
    for _ in range(replicates):
        order = rng.permutation(len(combined))
        randomized_left = profile_set(combined[order[:left_size]])
        randomized_right = profile_set(combined[order[left_size:]])
        for key in observed:
            value = float(jsd(randomized_left[key], randomized_right[key]))
            exceed[key] += value >= observed[key] - 1e-15
    return {
        key: {
            "jsd_bits": observed[key],
            "exceedances": exceed[key],
            "permutations": replicates,
            "plus_one_p": (1 + exceed[key]) / (replicates + 1),
        }
        for key in observed
    }


def permutation_mean_difference(
    left: np.ndarray, right: np.ndarray, replicates: int, rng: np.random.Generator
) -> dict[str, float | int]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    observed = float(left.mean() - right.mean())
    combined = np.concatenate([left, right])
    left_size = len(left)
    exceed = 0
    for _ in range(replicates):
        order = rng.permutation(len(combined))
        difference = float(
            combined[order[:left_size]].mean() - combined[order[left_size:]].mean()
        )
        exceed += abs(difference) >= abs(observed) - 1e-15
    return {
        "mean_difference": observed,
        "exceedances_two_sided": exceed,
        "permutations": replicates,
        "plus_one_p_two_sided": (1 + exceed) / (replicates + 1),
    }


def within_tier_audit(
    groups: dict[str, np.ndarray], replicates: int
) -> dict[str, object]:
    output: dict[str, object] = {}
    family_refs: list[tuple[str, str]] = []
    family_p: list[float] = []
    for tier_index, (tier, h_key, t_key) in enumerate(
        (("Bottom 5%", "HB", "TB"), ("Top 5%", "HT", "TT"))
    ):
        left = groups[h_key]
        right = groups[t_key]
        profiles = permutation_jsd(
            left,
            right,
            replicates,
            np.random.default_rng(LEGACY_SEED + 2000 + tier_index),
        )
        richness = permutation_mean_difference(
            (left > 0).sum(axis=1),
            (right > 0).sum(axis=1),
            replicates,
            np.random.default_rng(LEGACY_SEED + 2100 + tier_index),
        )
        output[tier] = {"richness": richness, "profiles": profiles}
        family_refs.append((tier, "richness"))
        family_p.append(float(richness["plus_one_p_two_sided"]))
        for mode in ("amount", "equal_wallet", "prevalence"):
            family_refs.append((tier, mode))
            family_p.append(float(profiles[mode]["plus_one_p"]))
    for (tier, name), adjusted in zip(family_refs, holm_adjust(family_p)):
        tier_record = output[tier]
        if name == "richness":
            tier_record["richness"]["holm_p_8"] = float(adjusted)
        else:
            tier_record["profiles"][name]["holm_p_8"] = float(adjusted)
    return output


def wallet_features(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, slice]]:
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1)
    proportions = matrix / totals[:, None]
    binary = (matrix > 0).astype(float)
    upper = np.triu_indices(matrix.shape[1], 1)
    amount_outer = (
        totals[:, None, None]
        * proportions[:, :, None]
        * proportions[:, None, :]
    )[:, upper[0], upper[1]]
    equal_outer = (proportions[:, :, None] * proportions[:, None, :])[
        :, upper[0], upper[1]
    ]
    binary_outer = (binary[:, :, None] * binary[:, None, :])[
        :, upper[0], upper[1]
    ]
    blocks = [matrix, proportions, binary, amount_outer, equal_outer, binary_outer]
    slices: dict[str, slice] = {}
    start = 0
    for name, block in zip(METRICS, blocks):
        slices[name] = slice(start, start + block.shape[1])
        start += block.shape[1]
    return np.concatenate(blocks, axis=1), slices


def observed_metric_jsd(
    left: np.ndarray, right: np.ndarray
) -> tuple[dict[str, float], dict[str, slice]]:
    combined = np.vstack([left, right])
    features, slices = wallet_features(combined)
    left_sum = features[: len(left)].sum(axis=0)
    right_sum = features[len(left) :].sum(axis=0)
    return (
        {
            name: float(jsd(left_sum[block], right_sum[block]))
            for name, block in slices.items()
        },
        slices,
    )


def random_group_sums(
    features: np.ndarray,
    selected_size: int,
    draws: int,
    rng: np.random.Generator,
    batch_size: int = 250,
):
    """Yield uniform fixed-size subset sums using independent random-key ranks."""
    rows = len(features)
    total = features.sum(axis=0)
    emitted = 0
    while emitted < draws:
        batch = min(batch_size, draws - emitted)
        keys = rng.random((batch, rows))
        selected = np.argpartition(keys, selected_size - 1, axis=1)[:, :selected_size]
        indicator = np.zeros((batch, rows), dtype=float)
        indicator[np.arange(batch)[:, None], selected] = 1.0
        left = indicator @ features
        yield left, total[None, :] - left
        emitted += batch


def direct_delta_audit(
    groups: dict[str, np.ndarray], replicates: int, seed: int
) -> dict[str, object]:
    bottom_observed, slices = observed_metric_jsd(groups["HB"], groups["TB"])
    top_observed, top_slices = observed_metric_jsd(groups["HT"], groups["TT"])
    if slices != top_slices:
        raise AssertionError("Metric coordinate slices differ across tiers")
    observed_delta = {
        name: bottom_observed[name] - top_observed[name] for name in METRICS
    }

    bottom_features, _ = wallet_features(np.vstack([groups["HB"], groups["TB"]]))
    top_features, _ = wallet_features(np.vstack([groups["HT"], groups["TT"]]))
    rng = np.random.default_rng(seed)
    bottom_draws = random_group_sums(
        bottom_features, len(groups["HB"]), replicates, rng
    )
    top_draws = random_group_sums(top_features, len(groups["HT"]), replicates, rng)
    null = {name: np.empty(replicates, dtype=float) for name in METRICS}
    offset = 0
    for (bottom_h, bottom_t), (top_h, top_t) in zip(bottom_draws, top_draws):
        batch = len(bottom_h)
        for name, block in slices.items():
            null[name][offset : offset + batch] = (
                jsd(bottom_h[:, block], bottom_t[:, block])
                - jsd(top_h[:, block], top_t[:, block])
            )
        offset += batch
    if offset != replicates:
        raise AssertionError("Incorrect number of permutation draws")

    records: dict[str, dict[str, float | int]] = {}
    primary_p: list[float] = []
    two_sided_p: list[float] = []
    for name in METRICS:
        values = null[name]
        observed = observed_delta[name]
        upper_exceed = int(np.sum(values >= observed - 1e-15))
        center = float(values.mean())
        two_exceed = int(np.sum(np.abs(values - center) >= abs(observed - center) - 1e-15))
        record = {
            "bottom_jsd_bits": bottom_observed[name],
            "top_jsd_bits": top_observed[name],
            "delta_bottom_minus_top": observed,
            "null_mean": center,
            "null_median": float(np.median(values)),
            "null_sd": float(values.std(ddof=1)),
            "null_q025": float(np.quantile(values, 0.025)),
            "null_q975": float(np.quantile(values, 0.975)),
            "upper_tail_exceedances": upper_exceed,
            "plus_one_p_directional": (1 + upper_exceed) / (replicates + 1),
            "centered_two_sided_exceedances": two_exceed,
            "plus_one_p_centered_two_sided": (1 + two_exceed) / (replicates + 1),
        }
        records[name] = record
        primary_p.append(float(record["plus_one_p_directional"]))
        two_sided_p.append(float(record["plus_one_p_centered_two_sided"]))
    for name, adjusted in zip(METRICS, holm_adjust(primary_p)):
        records[name]["holm_p_6_directional"] = float(adjusted)
    for name, adjusted in zip(METRICS, holm_adjust(two_sided_p)):
        records[name]["holm_p_6_centered_two_sided"] = float(adjusted)

    return {
        "statistic": "Delta = JSD(HB,TB) - JSD(HT,TT)",
        "directional_alternative": "Delta > 0",
        "permutations": replicates,
        "seed": seed,
        "randomization": (
            "Candidate labels are independently reassigned within Bottom and Top "
            "among fixed October-active wallet vectors, preserving observed active "
            "Harris/Trump group sizes in each tier. The same assignment is used for "
            "all six views within a replicate."
        ),
        "conditioning": (
            "The tier-specific wallet-vector multisets are fixed. Thus total spend, "
            "subject breadth, within-wallet allocation concentration, and all subject "
            "amounts are conditioned on rather than simulated. Exchangeability of "
            "candidate labels within tier is the identifying null assumption."
        ),
        "multiplicity": (
            "Holm family-wise adjustment across the six specified directional tests; "
            "centered two-sided probabilities are included as a sensitivity."
        ),
        "metrics": records,
    }


def format_p(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else "$<0.0001$"


def write_tables(output: dict[str, object], tex_path: Path, csv_path: Path) -> None:
    direct = output["direct_bottom_minus_top_test"]
    records = direct["metrics"]
    tex_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Representation & Bottom JSD & Top JSD & $\Delta$ & $p_{+}$ & Holm $p_{+}$ \\",
        r"\midrule",
    ]
    csv_rows = []
    for name in METRICS:
        record = records[name]
        tex_lines.append(
            f"{DISPLAY[name]} & {record['bottom_jsd_bits']:.3f} & "
            f"{record['top_jsd_bits']:.3f} & {record['delta_bottom_minus_top']:.3f} & "
            f"{format_p(record['plus_one_p_directional'])} & "
            f"{format_p(record['holm_p_6_directional'])} \\\\"
        )
        csv_rows.append(
            {
                "representation": name,
                "bottom_jsd_bits": record["bottom_jsd_bits"],
                "top_jsd_bits": record["top_jsd_bits"],
                "delta_bottom_minus_top": record["delta_bottom_minus_top"],
                "upper_tail_exceedances": record["upper_tail_exceedances"],
                "permutations": direct["permutations"],
                "plus_one_p_directional": record["plus_one_p_directional"],
                "holm_p_6_directional": record["holm_p_6_directional"],
                "plus_one_p_centered_two_sided": record["plus_one_p_centered_two_sided"],
                "holm_p_6_centered_two_sided": record["holm_p_6_centered_two_sided"],
            }
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"% Directional fixed-vector randomization test: Delta > 0; 9,999 permutations.",
        ]
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    args = parse_args()
    subjects, groups = load_vectors(args.input)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    expected_hash = metadata["released_csv"]["sha256"]
    observed_hash = sha256(args.input)
    if expected_hash != observed_hash:
        raise ValueError("Compact vector CSV hash does not match its metadata")

    within = within_tier_audit(groups, args.permutations)
    direct = direct_delta_audit(groups, args.permutations, args.direct_seed)
    output = {
        "analysis": "Election candidate-label permutation audit",
        "input": {
            "path": args.input.name,
            "sha256": observed_hash,
            "subjects": subjects,
            "active_group_sizes": {name: len(matrix) for name, matrix in groups.items()},
        },
        "within_tier_reproduction": {
            "purpose": (
                "Reproduction of the previously reported exploratory eight-test "
                "family; profile tests are upper-tail and richness tests are two-sided."
            ),
            "legacy_seed_base": LEGACY_SEED,
            "permutations": args.permutations,
            "results": within,
        },
        "direct_bottom_minus_top_test": direct,
        "limitations": [
            "The direct six-view randomization is a retrospective robustness audit, not a preregistered analysis.",
            "Inference is conditional on October activity and the fixed September cohort definitions.",
            "Wallet rows are treated as exchangeable and independent within purchase-size tier.",
            "The audit concerns one selected election event and one follow-up month.",
            "The test does not identify wealth, ideology, coordination, learning, or information advantage.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(output), indent=2), encoding="utf-8")
    write_tables(output, args.table, args.table_csv)
    print(json.dumps(clean(direct["metrics"]), indent=2))


if __name__ == "__main__":
    main()
