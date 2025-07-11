#!/bin/bash

set +e

EXPERIMENTS=(
  "experiments/fft-rd100"
  "experiments/ffa-rd100"
  "experiments/ffa-rd200"
  "experiments/ffa-rd300"
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
