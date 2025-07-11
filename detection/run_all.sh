#!/bin/bash

set +e

EXPERIMENTS=(
  "experiments/fft-rd10"
  "experiments/ffa-rd20"
  "experiments/ffa-rd30"
  "experiments/ffa-rd50"
  "experiments/ffa-rd80"
  "experiments/ffa-rd100"
  "experiments/ffa-rd150"
  "experiments/ffa-rd200"
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
