#!/bin/bash

# !Only Seed and fold variable will be used. The rest is just placeholder incase I need it next time!

seed=22
remap=True
threshold=0.0
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
