#!/bin/bash

APR=$1
FP=$2
DATA=$3
OUT=$4

python3 run_algos.py $APR $FP $DATA $OUT
python3 plot_runtime.py $OUT/times.txt $OUT/plot.png
