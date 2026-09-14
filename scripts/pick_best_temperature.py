#!/usr/bin/env python3
"""Select the best posterior temperature from a temperature-sweep CSV.

Usage: pick_best_temperature.py CSV_PATH RUN_LABEL MC_SAMPLES

Prints the temperature with the lowest mc_bma_nll at the given MC budget on
stdout and the full ranking on stderr. Tolerates stray repeated header rows
in concatenated CSVs.
"""
import csv
import sys


def _is_mc(row, best_mc):
    mc = row.get("mc_samples", "")
    return mc.isdigit() and int(mc) == best_mc


def main(argv):
    if len(argv) != 4:
        raise SystemExit(f"usage: {argv[0]} CSV_PATH RUN_LABEL MC_SAMPLES")
    csv_path, run_label, best_mc = argv[1], argv[2], int(argv[3])

    with open(csv_path, newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r.get("run_label") == run_label and _is_mc(r, best_mc)
        ]
    if not rows:
        raise SystemExit(
            f"no rows in {csv_path} with run_label={run_label} and mc_samples={best_mc}"
        )

    scores = sorted((float(r["mc_bma_nll"]), r["temperature"]) for r in rows)
    for nll, temperature in scores:
        print(f"  T={temperature}: mc_bma_nll@{best_mc} = {nll:.8f}", file=sys.stderr)
    print(scores[0][1])


if __name__ == "__main__":
    main(sys.argv)
