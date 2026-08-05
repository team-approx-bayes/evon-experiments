#!/usr/bin/env bash

# Helper script to run Laplace Approximation evaluation from scripts/ directory.

set -euo pipefail

# Navigate to the clip-finetuning directory relative to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../clip-finetuning"

exec ./run_laplace.sh "$@"
