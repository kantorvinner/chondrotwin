#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data}"

echo "Expected on-disk layout under ${DATA_ROOT}:"
echo "  ${DATA_ROOT}/oai_zib/{train,val,test}/*.npz   (volume, seg, grade, node_mask arrays)"
echo "  ${DATA_ROOT}/ski10/{train,test}/*.npz"
echo "OAI access:    https://nda.nih.gov/oai   (Data Use Agreement required)"
echo "OAI-ZIB masks: https://pubdata.zib.de"
echo "Each shard stores a single knee as a compressed .npz with the four named arrays."
