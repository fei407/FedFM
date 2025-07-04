#!/bin/bash

set +e

EXPERIMENTS=(
#  "experiments/ffa-sqrt"
#  "experiments/ffa_dr-sqrt"
#  "experiments/lora_svd-normal"
#  "experiments/ffa-normal"
#  "experiments/lora_svd-sqrt"
  "experiments/lora_zp-normal"
  "experiments/lora_zp-sqrt"
  "experiments/lora_nbias-normal"
  "experiments/lora_nbias-sqrt"
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
