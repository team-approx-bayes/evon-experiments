#!/usr/bin/env python3
"""
Retrieve Laplace evaluation results from WandB across seeds (e.g., 42 to 46) and generate
a Markdown table and CSV export comparing MAP vs Probit performance with mean ± std.
"""

import argparse
import csv
import io
import sys
from pathlib import Path
import numpy as np
import wandb


def parse_seeds(seed_str: str) -> list[int]:
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return sorted(list(set(seeds)))


def fetch_wandb_runs(project: str, entity: str | None = None):
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)
    return list(runs)


def extract_single_run_metrics(run):
    summary = run.summary or {}
    map_metrics = {}
    probit_metrics = {}

    try:
        hist = run.history(keys=["acc/test", "loss/test", "calibration_mean/ece", "calibration_mean/brier"])
        row0 = hist.iloc[0] if not hist.empty else {}
    except Exception:
        row0 = {}

    map_acc = row0.get("acc/test") if "acc/test" in row0 and not np.isnan(row0["acc/test"]) else summary.get("test_acc")
    map_loss = row0.get("loss/test") if "loss/test" in row0 and not np.isnan(row0["loss/test"]) else summary.get("test_loss")
    map_ece = row0.get("calibration_mean/ece") if "calibration_mean/ece" in row0 and not np.isnan(row0["calibration_mean/ece"]) else None
    map_brier = row0.get("calibration_mean/brier") if "calibration_mean/brier" in row0 and not np.isnan(row0["calibration_mean/brier"]) else None

    map_metrics["acc"] = map_acc
    map_metrics["loss"] = map_loss
    map_metrics["ece"] = map_ece
    map_metrics["brier"] = map_brier

    probit_metrics["acc"] = summary.get("probit/test_acc")
    probit_metrics["loss"] = summary.get("probit/test_loss")
    probit_metrics["ece"] = summary.get("probit/ece")
    probit_metrics["brier"] = summary.get("probit/brier")

    return map_metrics, probit_metrics


def extract_metrics(runs: list, target_seeds: list[int] | None = None):
    """
    Extract MAP and Probit metrics from wandb runs grouped by dataset_name and seed.
    Returns: dict[ds][seed] = {"map": map_metrics, "probit": probit_metrics}
    """
    latest_runs = {}

    for run in runs:
        ds = run.config.get("dataset_name")
        if not ds:
            continue

        raw_seed = run.config.get("seed")
        if raw_seed is None:
            continue
        try:
            seed = int(raw_seed)
        except (ValueError, TypeError):
            continue

        if target_seeds is not None and seed not in target_seeds:
            continue

        key = (ds, seed)
        if key not in latest_runs or getattr(run, "updated", "") > getattr(latest_runs[key], "updated", ""):
            latest_runs[key] = run

    data_by_dataset = {}
    for (ds, seed), run in latest_runs.items():
        if ds not in data_by_dataset:
            data_by_dataset[ds] = {}
        map_m, probit_m = extract_single_run_metrics(run)
        data_by_dataset[ds][seed] = {
            "map": map_m,
            "probit": probit_m,
        }

    return data_by_dataset


def format_mean_std(vals: list[float], decs: int, is_delta: bool = False, bold: bool = False) -> str:
    if not vals:
        return "-"
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    sign = "+" if (is_delta and mean_val > 0) else ""

    if len(vals) > 1:
        text = f"{sign}{mean_val:.{decs}f} ± {std_val:.{decs}f}"
    else:
        text = f"{sign}{mean_val:.{decs}f}"

    if bold:
        return f"**{text}**"
    return text


def generate_table_rows(data_by_dataset: dict) -> tuple[list[str], list[list[str]]]:
    datasets = sorted(data_by_dataset.keys())
    if not datasets:
        return [], []

    headers = ["Metric", "Method"] + datasets + ["Avg"]

    metrics_info = [
        ("Accuracy (%) ↑", "acc", True, 2),
        ("Loss ↓", "loss", False, 4),
        ("ECE (x100) ↓", "ece", True, 2),
        ("Brier Score ↓", "brier", False, 4),
    ]

    all_seeds = set()
    for ds in datasets:
        all_seeds.update(data_by_dataset[ds].keys())
    sorted_seeds = sorted(all_seeds)

    md_rows = []
    csv_rows = [headers]

    for metric_name, key, is_pct, decs in metrics_info:
        mult = 100.0 if is_pct else 1.0

        map_cells = []
        probit_cells = []
        delta_cells = []

        csv_map_cells = []
        csv_probit_cells = []
        csv_delta_cells = []

        map_seed_avgs = {s: [] for s in sorted_seeds}
        probit_seed_avgs = {s: [] for s in sorted_seeds}
        delta_seed_avgs = {s: [] for s in sorted_seeds}

        for ds in datasets:
            m_vals = []
            p_vals = []
            d_vals = []

            for s in sorted_seeds:
                seed_data = data_by_dataset[ds].get(s)
                if not seed_data:
                    continue
                m_val = seed_data["map"].get(key)
                p_val = seed_data["probit"].get(key)

                if m_val is not None:
                    m_scaled = m_val * mult
                    m_vals.append(m_scaled)
                    map_seed_avgs[s].append(m_scaled)

                if p_val is not None:
                    p_scaled = p_val * mult
                    p_vals.append(p_scaled)
                    probit_seed_avgs[s].append(p_scaled)

                if m_val is not None and p_val is not None:
                    d_scaled = (p_val - m_val) * mult
                    d_vals.append(d_scaled)
                    delta_seed_avgs[s].append(d_scaled)

            map_cells.append(format_mean_std(m_vals, decs))
            probit_cells.append(format_mean_std(p_vals, decs))
            delta_cells.append(format_mean_std(d_vals, decs, is_delta=True, bold=True))

            csv_map_cells.append(format_mean_std(m_vals, decs) if m_vals else "")
            csv_probit_cells.append(format_mean_std(p_vals, decs) if p_vals else "")
            csv_delta_cells.append(format_mean_std(d_vals, decs, is_delta=True, bold=False) if d_vals else "")

        # Calculate Avg across datasets per seed, then aggregate across seeds (mean only in Avg column)
        m_avg_vals = [np.mean(map_seed_avgs[s]) for s in sorted_seeds if map_seed_avgs[s]]
        p_avg_vals = [np.mean(probit_seed_avgs[s]) for s in sorted_seeds if probit_seed_avgs[s]]
        d_avg_vals = [np.mean(delta_seed_avgs[s]) for s in sorted_seeds if delta_seed_avgs[s]]

        def format_mean_only(vals: list[float], decs: int, is_delta: bool = False, bold: bool = False) -> str:
            if not vals:
                return "-"
            mean_val = float(np.mean(vals))
            sign = "+" if (is_delta and mean_val > 0) else ""
            text = f"{sign}{mean_val:.{decs}f}"
            if bold:
                return f"**{text}**"
            return text

        mean_m = format_mean_only(m_avg_vals, decs)
        mean_p = format_mean_only(p_avg_vals, decs)
        mean_d = format_mean_only(d_avg_vals, decs, is_delta=True, bold=True)

        csv_mean_m = format_mean_only(m_avg_vals, decs) if m_avg_vals else ""
        csv_mean_p = format_mean_only(p_avg_vals, decs) if p_avg_vals else ""
        csv_mean_d = format_mean_only(d_avg_vals, decs, is_delta=True, bold=False) if d_avg_vals else ""

        md_rows.append(f"| **{metric_name}** | MAP | " + " | ".join(map_cells) + f" | {mean_m} |")
        md_rows.append(f"| | Probit | " + " | ".join(probit_cells) + f" | {mean_p} |")
        md_rows.append(f"| | Δ (Probit - MAP) | " + " | ".join(delta_cells) + f" | {mean_d} |")

        csv_rows.append([metric_name, "MAP"] + csv_map_cells + [csv_mean_m])
        csv_rows.append(["", "Probit"] + csv_probit_cells + [csv_mean_p])
        csv_rows.append(["", "Δ (Probit - MAP)"] + csv_delta_cells + [csv_mean_d])

    return md_rows, csv_rows


def generate_markdown_table(data_by_dataset: dict) -> str:
    datasets = sorted(data_by_dataset.keys())
    if not datasets:
        return "No run data found."

    headers = ["Metric", "Method"] + datasets + ["Avg"]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join([":---", ":---"] + [":---:"] * (len(datasets) + 1)) + " |"

    md_rows, _ = generate_table_rows(data_by_dataset)
    return "\n".join([header_row, separator_row] + md_rows)


def generate_csv_data(data_by_dataset: dict) -> str:
    datasets = sorted(data_by_dataset.keys())
    if not datasets:
        return ""

    _, csv_rows = generate_table_rows(data_by_dataset)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(csv_rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Fetch WandB Laplace evaluation runs and output Markdown/CSV table with std across seeds.")
    parser.add_argument("--wandb-project", default="laplace-clip", help="WandB project name")
    parser.add_argument("--wandb-entity", default=None, help="WandB entity/username")
    parser.add_argument("--seeds", default="42-46", help="Comma-separated list or range of seeds (e.g. 42-46 or 42,43,44,45,46)")
    parser.add_argument("--output", default=None, help="Path to save markdown table file")
    parser.add_argument("--csv-output", default=None, help="Path to save CSV results file")

    args = parser.parse_args()

    target_seeds = parse_seeds(args.seeds)
    runs = fetch_wandb_runs(project=args.wandb_project, entity=args.wandb_entity)
    print(f"[wandb] Found {len(runs)} total runs in project '{args.wandb_project}' (filtering seeds: {target_seeds})", file=sys.stdout)

    data_by_dataset = extract_metrics(runs, target_seeds=target_seeds)
    table_md = generate_markdown_table(data_by_dataset)
    table_csv = generate_csv_data(data_by_dataset)

    print("\n" + table_md + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(table_md + "\n", encoding="utf-8")
        print(f"[file] Markdown table saved to {out_path}", file=sys.stdout)

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.write_text(table_csv, encoding="utf-8")
        print(f"[file] CSV table saved to {csv_path}", file=sys.stdout)


if __name__ == "__main__":
    main()

