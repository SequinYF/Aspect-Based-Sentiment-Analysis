#!/bin/bash

batch=32
epochs=10
lr=0.001
lr_schedule_values=("False", "True")
adapter_values=("True" "False")

for lr_schedule in "${lr_schedule_values[@]}"; do
    for adapter in "${adapter_values[@]}"; do
        echo "Running ABTE training with lr_schedule=$lr_schedule and adapter=$adapter"
        python -m train_ABSA ABTE --batch="$batch" --epochs="$epochs" --lr="$lr" --lr_schedule="$lr_schedule" --adapter="$adapter"
        echo "--------------------"
    done
done

for lr_schedule in "${lr_schedule_values[@]}"; do
    for adapter in "${adapter_values[@]}"; do
        echo "Running ABSA training with lr_schedule=$lr_schedule and adapter=$adapter"
        python -m train_ABSA ABSA --batch="$batch" --epochs="$epochs" --lr="$lr" --lr_schedule="$lr_schedule" --adapter="$adapter"
        echo "--------------------"
    done
done