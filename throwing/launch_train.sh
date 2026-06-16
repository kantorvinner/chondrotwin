#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
NPROC="${NPROC:-4}"

torchrun --standalone --nproc_per_node="${NPROC}" \
    -m chondrotwin.gallery.train +experiment="${EXPERIMENT}"
