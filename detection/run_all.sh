#!/bin/bash

set +e

EXPERIMENTS=(
  "experiments/ffa_le1-dr"
  "experiments/ffa_le2"
  "experiments/ffa_le3"
)

for exp_dir in "${EXPERIMENTS[@]}"
do
  echo "▶️ Running: $exp_dir"
  flwr run "$exp_dir" --stream

  if [ $? -ne 0 ]; then
    echo "❌ Experiment failed: $exp_dir"
  else
    echo "✅ Finished: $exp_dir"
  fi
done
