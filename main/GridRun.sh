#!/bin/bash

idx=0
LOG_FILE="./tmp/log/$(date +"%Y%m%d%H%M%S")"

for w in 4 8 16; do
	for nl in 2 4; do
		for ins in 128 256; do
			for nh in 2 4; do
				for cw in True False; do					
					echo "Running job $idx" >> $LOG_FILE
					python GridSearch.py \
						   --window_size $w \
						   --n_layers $nl \
						   --internal_size $ins \
						   --n_heads $nh \
						   --class_weight $cw \
						   >> $LOG_FILE 2>&1
					
					idx=$((idx+1))
					sleep 1
				done
			done
		done
	done
done


