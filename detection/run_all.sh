#!/bin/bash

set +e

EXPERIMENTS=(
  "experiments/ffa_lr5e-4"
  "experiments/ffa_lr1e-4"
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
