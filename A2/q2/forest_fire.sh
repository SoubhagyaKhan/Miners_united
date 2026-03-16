#!/bin/bash

GRAPH=$1
SEEDS=$2
OUT=$3
K=$4
R=$5
HOPS=$6

python3 -m forest_fire $GRAPH $SEEDS $OUT $K $R $HOPS