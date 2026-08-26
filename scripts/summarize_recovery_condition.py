#!/usr/bin/env python3
"""Report the conditional Proposition 2 sufficient-certificate rate.

This script is intentionally analysis-only: it does not import or run AURA,
an optimizer, a simulator, or a robot interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results/error_experiment/proposition2_step_metrics.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results/error_experiment"
NUMERICAL_TOLERANCE = 1.0e-9
METRIC_TOLERANCE = 1.0e-12

CONDITION_ORDER = (
    ("double_integrator", "gaussian"),
    ("kinematic_car", "gaussian"),
    ("pushing_object", "gaussian"),
    ("kinematic_car", "mujoco"),
    ("pushing_object", "mujoco"),
)
EXPECTED_STEPS_PER_TRIAL = {
    ("double_integrator", "gaussian"): 10,
    ("kinematic_car", "gaussian"): 10,
    ("pushing_object", "gaussian"): 10,
    ("kinematic_car", "mujoco"): 10,
    ("pushing_object", "mujoco"): 10,
}
REQUIRED_COLUMNS = {
    "system",
    "environment",
    "condition",
    "trial",
    "step",
    "delta",
    "delta_source",
    "delta_kind",
    "x_target",
    "x_pred_nominal",
    "x_pred_optimized",
    "x_executed",
    "e_nominal",
    "e_optimized",
    "e_execution",
    "e_final",
}

SUMMARY_COLUMNS = (
    "system_environment",
    "certificate_given_execution_bound",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the saved Proposition 2 per-step measurements."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_vector(value: str) -> list[float]:
    vector = json.loads(value)
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"invalid state vector: {value!r}")
    parsed = [float(component) for component in vector]
    if not all(math.isfinite(component) for component in parsed):
        raise ValueError(f"non-finite state vector: {value!r}")
    return parsed


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def state_distance(system: str, first: list[float], second: list[float]) -> float:
    """Match the original experiment's ``utils.utils.arrayDistance`` metric."""

    if system == "double_integrator":
        if len(first) < 6 or len(second) < 6:
            raise ValueError("double-integrator states must have six components")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(first[:6], second[:6])))
    if system in {"kinematic_car", "pushing_object"}:
        if len(first) < 3 or len(second) < 3:
            raise ValueError("SE(2) states must have three components")
        position = math.hypot(first[0] - second[0], first[1] - second[1])
        orientation = abs(wrap_angle(first[2] - second[2]))
        # OMPL SE2StateSpace.distance uses unit weight for R2 and 0.5 for SO2.
        return position + 0.5 * orientation
    raise ValueError(f"unsupported system in saved source: {system}")


def read_and_audit(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"source is missing columns: {sorted(missing_columns)}")
        raw_rows = list(reader)

    evaluated: list[dict[str, Any]] = []
    missing_records: list[str] = []
    maximum_metric_difference = 0.0
    delta_by_condition: dict[tuple[str, str], set[float]] = defaultdict(set)
    rows_by_trial: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    metric_pairs = {
        "e_nominal": ("x_pred_nominal", "x_target"),
        "e_optimized": ("x_pred_optimized", "x_target"),
        "e_execution": ("x_executed", "x_pred_optimized"),
        "e_final": ("x_executed", "x_target"),
    }
    for source_index, raw in enumerate(raw_rows, start=2):
        identity = (
            f"line {source_index} ({raw.get('condition', '?')}, "
            f"trial={raw.get('trial', '?')}, step={raw.get('step', '?')})"
        )
        try:
            system = raw["system"]
            environment = raw["environment"]
            delta = float(raw["delta"])
            if not math.isfinite(delta) or delta <= 0.0:
                raise ValueError(f"invalid delta {delta}")
            metrics = {
                name: float(raw[name])
                for name in ("e_nominal", "e_optimized", "e_execution", "e_final")
            }
            if not all(math.isfinite(value) and value >= 0.0 for value in metrics.values()):
                raise ValueError("one or more error metrics are missing or non-finite")
            vectors = {
                name: parse_vector(raw[name])
                for name in {
                    vector_name
                    for pair in metric_pairs.values()
                    for vector_name in pair
                }
            }
            for metric_name, (first_name, second_name) in metric_pairs.items():
                recomputed = state_distance(
                    system, vectors[first_name], vectors[second_name]
                )
                difference = abs(recomputed - metrics[metric_name])
                maximum_metric_difference = max(maximum_metric_difference, difference)
                if difference > METRIC_TOLERANCE:
                    raise ValueError(
                        f"{metric_name} differs from the original norm by {difference:.3g}"
                    )
            trial = str(raw["trial"])
            step = int(raw["step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            missing_records.append(f"{identity}: {error}")
            continue

        row = dict(raw)
        row.update(metrics)
        row["delta"] = delta
        row["execution_bound_satisfied"] = (
            metrics["e_execution"] <= delta + NUMERICAL_TOLERANCE
        )
        row["certificate_satisfied"] = (
            row["execution_bound_satisfied"]
            and metrics["e_optimized"] + metrics["e_execution"]
            <= delta + NUMERICAL_TOLERANCE
        )
        evaluated.append(row)
        key = (system, environment)
        delta_by_condition[key].add(delta)
        rows_by_trial[(system, environment, trial)].append(step)

    inconsistent_deltas = {
        key: sorted(values)
        for key, values in delta_by_condition.items()
        if len(values) != 1
    }
    if inconsistent_deltas:
        raise ValueError(f"inconsistent delta within condition: {inconsistent_deltas}")

    incomplete_trials: list[str] = []
    for (system, environment, trial), steps in rows_by_trial.items():
        expected_count = EXPECTED_STEPS_PER_TRIAL[(system, environment)]
        expected_steps = list(range(expected_count))
        actual_steps = sorted(steps)
        if actual_steps != expected_steps:
            incomplete_trials.append(
                f"{system}/{environment} trial {trial}: expected {expected_steps}, "
                f"found {actual_steps}"
            )

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    audit = {
        "source_records": len(raw_rows),
        "evaluated_steps": len(evaluated),
        "missing_records": missing_records,
        "incomplete_trials": incomplete_trials,
        "maximum_metric_recalculation_difference": maximum_metric_difference,
        "source_sha256": source_hash,
    }
    if missing_records or incomplete_trials:
        details = "\n".join(missing_records + incomplete_trials)
        raise ValueError(f"source audit failed:\n{details}")
    return evaluated, audit


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def count_text(numerator: int, denominator: int) -> str:
    pct = percentage(numerator, denominator)
    return f"{numerator} / {denominator} ({pct:.1f}%)" if denominator else "N/A"


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    combined: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty condition")

    eligible = [row for row in rows if row["execution_bound_satisfied"]]
    successful = [row for row in eligible if row["certificate_satisfied"]]
    if not eligible:
        raise ValueError("no steps satisfy the execution-error assumption")

    trials = {
        (row["system"], row["environment"], str(row["trial"])) for row in rows
    }
    first = rows[0]
    if combined:
        label = "All evaluated conditions"
    else:
        label = first["condition"]

    n_total = len(rows)
    n_eligible = len(eligible)
    n_success = len(successful)
    return {
        "system_environment": label,
        "system": "all" if combined else first["system"],
        "environment": "all" if combined else first["environment"],
        "trials": len(trials),
        "evaluated_steps": n_total,
        "eligible_steps": n_eligible,
        "excluded_steps": n_total - n_eligible,
        "certificate_steps": n_success,
        "certificate_pct": percentage(n_success, n_eligible),
        "certificate_given_execution_bound": count_text(n_success, n_eligible),
        "delta": first["delta"] if not combined else "condition-specific",
        "delta_source": first["delta_source"] if not combined else "See source rows",
        "delta_kind": first["delta_kind"] if not combined else "mixed",
    }


def build_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["environment"])].append(row)
    unexpected = set(grouped).difference(CONDITION_ORDER)
    missing = set(CONDITION_ORDER).difference(grouped)
    if unexpected or missing:
        raise ValueError(
            f"condition mismatch; unexpected={sorted(unexpected)}, missing={sorted(missing)}"
        )
    summaries = [summarize_rows(grouped[key]) for key in CONDITION_ORDER]
    summaries.append(summarize_rows(rows, combined=True))
    return summaries


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {field: summary[field] for field in SUMMARY_COLUMNS}
            for summary in summaries
        )


def latex_label(label: str) -> str:
    return (
        label.replace("—", "--")
        .replace("Real World", "Real-World Hardware")
        .replace("All evaluated conditions", "All conditions")
    )


def write_latex(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical evaluation of the sufficient approximate-recovery condition in Proposition~2. A step is evaluated only when $e_{\mathrm{exec}}\leq\Delta$, and the certificate is satisfied when $e_{\mathrm{opt}}+e_{\mathrm{exec}}\leq\Delta$.}",
        r"\label{tab:recovery-condition-validation}",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"System / Environment & $e_{\mathrm{opt}}+e_{\mathrm{exec}}\leq\Delta$"
        + " " + chr(92) * 2,
        r"\midrule",
    ]
    for index, summary in enumerate(summaries):
        if index == len(summaries) - 1:
            lines.append(r"\midrule")
        lines.append(
            "{} & {} \\\\".format(
                latex_label(str(summary["system_environment"])),
                summary["certificate_given_execution_bound"].replace("%", r"\%"),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def interpretation(summary: dict[str, Any]) -> str:
    return (
        f"For {summary['system_environment']}, {summary['eligible_steps']}/"
        f"{summary['evaluated_steps']} steps satisfied e_exec <= Delta. Among "
        f"only those eligible steps, e_opt + e_exec <= Delta in "
        f"{summary['certificate_steps']}/{summary['eligible_steps']} steps "
        f"({summary['certificate_pct']:.1f}%)."
    )


def write_text(
    path: Path,
    source: Path,
    summaries: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    lines = [
        "Proposition 2 conditional empirical evaluation",
        "",
        "Reported test",
        "- First evaluate e_exec = d(x_executed, Gamma(x_current, u_optimized)).",
        "- Exclude steps for which e_exec > Delta.",
        "- On the remaining steps, the sufficient certificate is e_opt + e_exec <= Delta.",
        "- The CSV and LaTeX table contain only this conditional certificate result.",
        "",
        "Source data",
        f"- Working-tree source: {source.resolve()}",
        "- All five simulation conditions were rerun by experiment/recovery_condition.py.",
        "- The real-world row was removed from this analysis.",
        f"- SHA-256: {audit['source_sha256']}",
        "- No robot execution was performed.",
        "",
        "Consistency audit",
        f"- Source records: {audit['source_records']}",
        f"- Evaluated execution steps: {audit['evaluated_steps']}",
        f"- Distinct trials/trajectories: {summaries[-1]['trials']}",
        f"- Failed or incomplete included trials: {len(audit['incomplete_trials'])}",
        f"- Missing/invalid records: {len(audit['missing_records'])}",
        f"- Incomplete trial sequences: {len(audit['incomplete_trials'])}",
        f"- Maximum absolute difference when recalculating all four errors: {audit['maximum_metric_recalculation_difference']:.3g}",
        "- Counts are per execution step, not per trial.",
        "- The denominator contains only steps satisfying e_exec <= Delta with a 1e-9 numerical tolerance.",
        "- Certificate evaluation uses e_opt + e_exec <= Delta with a 1e-9 numerical tolerance.",
        "- Double-integrator errors use the Euclidean R^6 norm.",
        "- Car and pushing errors use OMPL SE2StateSpace.distance: Euclidean XY distance plus 0.5 times the wrapped absolute yaw difference.",
        "",
        "Delta definitions",
    ]
    grouped = {
        (row["system"], row["environment"]): row
        for row in summaries[:-1]
    }
    for key in CONDITION_ORDER:
        summary = grouped[key]
        lines.append(
            f"- {summary['system_environment']}: Delta={float(summary['delta']):.15g}; "
            f"{summary['delta_kind']}; {summary['delta_source']}."
        )
    lines.extend(["", "Results"])
    lines.extend(f"- {interpretation(summary)}" for summary in summaries)
    lines.extend(
        [
            "",
            "Important limitation",
            "Every Delta is an empirical recovery-tube radius frozen from a disjoint 100-trial calibration set by adding the separately observed maximum optimization and execution budgets. These empirical radii are not universal mathematical bounds for every possible disturbance or physics state.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"saved per-step source not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, audit = read_and_audit(source)
    summaries = build_summaries(rows)
    csv_path = output_dir / "proposition2_summary.csv"
    latex_path = output_dir / "proposition2_table.tex"
    text_path = output_dir / "proposition2_summary.txt"
    write_csv(csv_path, summaries)
    write_latex(latex_path, summaries)
    write_text(text_path, source, summaries, audit)

    for summary in summaries:
        print(interpretation(summary))
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")
    print(f"Summary: {text_path}")


if __name__ == "__main__":
    main()
