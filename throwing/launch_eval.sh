#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"

python -m chondrotwin.gallery.evaluate +experiment="${EXPERIMENT}"
python -m chondrotwin.gallery.panel_uncertainty +experiment="${EXPERIMENT}"
