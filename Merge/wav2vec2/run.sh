#!/bin/bash
epochs=100 #3
early_stopping=20
batch_size=8
lr=1e-5
seed=22

set -ue

for fold in {0..9}
do
    echo "============training seed $seed============"
   python ./train.py \
        --epochs $epochs \
        --earlystopping $early_stopping \
        --batch-size $batch_size \
        --lr $lr \
        --seed $seed \
        --fold $fold \

done
