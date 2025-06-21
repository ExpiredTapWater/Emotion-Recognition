#!/bin/bash

seed=22
remap=True
threshold=0.7
mode='simple'
flip=False

set -ue

for fold in {0..9}
do
   echo "============ Fold $fold ============"
   python ./evaluate.py \
        --seed $seed \
	--remap $remap \
	--threshold $threshold \
	--flip $flip \
	--mode $mode \
        --fold $fold \

done
