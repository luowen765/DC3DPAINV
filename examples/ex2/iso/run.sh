#!/usr/bin/env bash
set -euo pipefail

NP=${NP:-32}
make -C .. DC3DInv
mpirun -np "${NP}" ../DC3DInv inversion_parameters.txt 2>&1 | tee run.log
