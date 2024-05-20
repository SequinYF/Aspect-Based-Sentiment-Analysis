#!/bin/bash
# Usage: ./train.sh ABTE


batch=8
epochs=5
lr_schedule_values=("False", "True")
adapter_values=("True" "False")


if [ $# -eq 0 ]; then
    echo "Please provide a model name as an argument."
    echo "Usage: $0 <model_name>"
    echo "Available models: ABTE, ABSA"
    exit 1
fi

model=$1

if [ "$model" == "ABTE" ]; then
    lr=3*1e-5
else
    lr=1e-5
fi


for lr_schedule in "${lr_schedule_values[@]}"; do
    for adapter in "${adapter_values[@]}"; do
        echo "Running $model training with lr_schedule=$lr_schedule and adapter=$adapter"
        python -m train_ABSA $model --batch="$batch" --epochs="$epochs" --lr="$lr" --lr_schedule="$lr_schedule" --adapter="$adapter"
        echo "--------------------"
    done
done