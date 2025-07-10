#!/bin/bash

set +e

EXPERIMENTS=(
  "experiments/ffa-rd100-cs0.2"
  "experiments/ffa-rd100-cs0.4"
  "experiments/ffa-rd100-cs0.6"
  "experiments/ffa-rd200-cs0.2"
  "experiments/ffa-rd200-cs0.4"
  "experiments/ffa-rd200-cs0.6"
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
