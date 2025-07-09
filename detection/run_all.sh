#!/bin/bash

set +e

EXPERIMENTS=(
  "experiments/ffa"
  "experiments/ffa_dr"
  "experiments/lora_zp"
  "experiments/lora_svd"
  "experiments/lora_nbias"
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
