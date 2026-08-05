#!/usr/bin/env python3
"""
Combine finetuning results (results_ft.csv) with Laplace Probit results (results_laplace.csv),
inserting a 'BayesVLM' row into each metric section, highlighting the best result for each column,
and outputting formatted Markdown and CSV tables.
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np

# Dataset mapping between laplace dataset names and ft column names
DATASET_MAP = {
    "cars": "CARS",
    "dtd": "DTD",
    "eurosat": "E-SAT",
    "gtsrb": "GTSRB",
    "mnist": "MNIST",
    "resisc45": "R45",
    "sun397": "SUN",
    "svhn": "SVHN",
    "Avg": "Avg",
}

# Metric mapping between laplace metric names and ft metric section names
METRIC_MAP = {
    "Accuracy (%) ↑": ("Acc. ↑", 2),
    "Loss ↓": ("NLL ↓", 3),
    "ECE (x100) ↓": ("ECE (%) ↓", 2),
    "Brier Score ↓": ("Brier ↓", 3),
}


def extract_num_and_std(val_str: str) -> tuple[float | None, float | None]:
    if not val_str or val_str == "-":
        return None, None
    clean = val_str.replace("*", "").strip()
    if "±" in clean:
        parts = clean.split("±")
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return None, None
    else:
        try:
            return float(clean), None
        except ValueError:
            return None, None


def format_val_std_str(val_str: str, decs: int) -> str:
    m, s = extract_num_and_std(val_str)
    if m is None:
        return "-"
    if s is not None:
        return f"{m:.{decs}f} ± {s:.{decs}f}"
    return f"{m:.{decs}f}"


def load_laplace_probit_values(laplace_csv_path: Path) -> dict:
    """
    Parse results_laplace.csv and extract Probit row values (including mean ± std) for each metric.
    Returns dict: { ft_metric_name: { dataset_col: str_val } }
    """
    laplace_data = {}
    current_metric_raw = None

    with open(laplace_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_raw = row.get("Metric", "").strip()
            if metric_raw:
                current_metric_raw = metric_raw

            method = row.get("Method", "").strip()

            if method != "Probit" or not current_metric_raw or current_metric_raw not in METRIC_MAP:
                continue

            ft_metric_name, decs = METRIC_MAP[current_metric_raw]
            laplace_data[ft_metric_name] = {}

            for laplace_ds, ft_ds in DATASET_MAP.items():
                if laplace_ds in row and row[laplace_ds]:
                    laplace_data[ft_metric_name][ft_ds] = row[laplace_ds].strip()

    return laplace_data


def extract_num(val_str: str) -> float | None:
    m, _ = extract_num_and_std(val_str)
    return m


def merge_and_generate_tables(ft_csv_path: Path, laplace_csv_path: Path, add_mean_col: bool = True) -> tuple[str, list[list[str]]]:
    laplace_probit = load_laplace_probit_values(laplace_csv_path)

    with open(ft_csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        return "Empty FT CSV file.", []

    header = reader[0]
    dataset_cols = header[2:]

    output_rows = []
    current_metric = None

    i = 1
    while i < len(reader):
        row = reader[i]
        if not row or not any(row):
            i += 1
            continue

        metric_col, opt_col = row[0].strip(), row[1].strip()

        if metric_col:
            current_metric = metric_col

        output_rows.append((current_metric, opt_col, row[2:]))

        # Insert BayesVLM row right after SOAP in each metric block
        if opt_col == "SOAP" and current_metric in laplace_probit:
            bayesvlm_vals = []
            probit_dict = laplace_probit[current_metric]

            if "Acc" in current_metric:
                decs = 2
            elif "NLL" in current_metric or "Brier" in current_metric:
                decs = 3
            else:
                decs = 2

            for ds in dataset_cols:
                if ds in probit_dict:
                    bayesvlm_vals.append(format_val_std_str(probit_dict[ds], decs))
                else:
                    bayesvlm_vals.append("-")

            output_rows.append((current_metric, "BayesVLM", bayesvlm_vals))

        i += 1

    # Group rows by metric
    blocks = []
    curr_metric = None
    curr_block = []

    for metric, opt, vals in output_rows:
        if metric != curr_metric:
            if curr_block:
                blocks.append((curr_metric, curr_block))
            curr_metric = metric
            curr_block = []
        curr_block.append((opt, vals))

    if curr_block:
        blocks.append((curr_metric, curr_block))

    md_lines = []
    csv_rows = []

    md_headers = ["Metric", "Optimizer"] + dataset_cols
    if add_mean_col:
        md_headers.append("Avg")

    header_line = "| " + " | ".join(md_headers) + " |"
    sep_line = "| " + " | ".join([":---", ":---"] + [":---:"] * (len(dataset_cols) + (1 if add_mean_col else 0))) + " |"

    md_lines.extend([header_line, sep_line])
    csv_rows.append(md_headers)

    for metric, rows in blocks:
        higher_is_better = "↑" in metric or "Acc" in metric

        full_rows = []
        num_matrix = []

        for opt, vals in rows:
            row_vals = list(vals)

            if add_mean_col:
                decs = 2 if ("Acc" in metric or "ECE" in metric) else 3
                if opt == "BayesVLM" and metric in laplace_probit and "Avg" in laplace_probit[metric]:
                    m = extract_num(laplace_probit[metric]["Avg"])
                    mean_str = f"{m:.{decs}f}" if m is not None else "-"
                else:
                    valid_nums = [extract_num(v) for v in vals if extract_num(v) is not None]
                    if valid_nums:
                        mean_str = f"{np.mean(valid_nums):.{decs}f}"
                    else:
                        mean_str = "-"
                row_vals.append(mean_str)

            full_rows.append((opt, row_vals))
            row_nums = [extract_num(v) for v in row_vals]
            num_matrix.append(row_nums)

        num_rows_cnt = len(full_rows)
        num_cols_cnt = len(full_rows[0][1])

        # Find best row indices for each column
        best_rows_per_col = {}
        for c in range(num_cols_cnt):
            col_vals = [num_matrix[r][c] for r in range(num_rows_cnt) if num_matrix[r][c] is not None]
            if not col_vals:
                best_rows_per_col[c] = set()
                continue

            target_best = max(col_vals) if higher_is_better else min(col_vals)

            best_set = set()
            for r in range(num_rows_cnt):
                v = num_matrix[r][c]
                if v is not None and abs(v - target_best) < 1e-5:
                    best_set.add(r)
            best_rows_per_col[c] = best_set

        # Build Markdown & CSV rows
        for r, (opt, row_vals) in enumerate(full_rows):
            display_metric = f"**{metric}**" if r == 0 else ""

            formatted_row_vals = []
            csv_formatted_row_vals = []

            for c, val_str in enumerate(row_vals):
                if r in best_rows_per_col[c] and val_str != "-":
                    formatted_row_vals.append(f"**{val_str}**")
                    csv_formatted_row_vals.append(f"**{val_str}**")
                else:
                    formatted_row_vals.append(val_str)
                    csv_formatted_row_vals.append(val_str)

            md_lines.append(f"| {display_metric} | {opt} | " + " | ".join(formatted_row_vals) + " |")

            csv_metric_label = metric if r == 0 else ""
            csv_rows.append([csv_metric_label, opt] + csv_formatted_row_vals)

    return "\n".join(md_lines), csv_rows


def main():
    parser = argparse.ArgumentParser(description="Merge results_ft.csv and results_laplace.csv highlighting best results per column.")
    parser.add_argument("--ft-csv", default="results_ft.csv", help="Path to results_ft.csv")
    parser.add_argument("--laplace-csv", default="results_laplace.csv", help="Path to results_laplace.csv")
    parser.add_argument("--output", default=None, help="Path to save merged Markdown table file")
    parser.add_argument("--csv-output", default=None, help="Path to save merged CSV file")
    parser.add_argument("--no-mean", action="store_true", help="Do not append Mean column")

    args = parser.parse_args()

    ft_path = Path(args.ft_csv)
    laplace_path = Path(args.laplace_csv)

    if not ft_path.exists():
        print(f"Error: {ft_path} not found.", file=sys.stderr)
        sys.exit(1)
    if not laplace_path.exists():
        print(f"Error: {laplace_path} not found.", file=sys.stderr)
        sys.exit(1)

    md_table, csv_rows = merge_and_generate_tables(ft_path, laplace_path, add_mean_col=not args.no_mean)

    print("\n" + md_table + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(md_table + "\n", encoding="utf-8")
        print(f"[file] Merged Markdown table saved to {out_path}", file=sys.stdout)

    if args.csv_output:
        csv_path = Path(args.csv_output)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"[file] Merged CSV saved to {csv_path}", file=sys.stdout)


if __name__ == "__main__":
    main()
