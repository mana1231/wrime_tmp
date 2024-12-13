#!/bin/bash
dt_start=$( date +"%s%3N" )

uv run train_bert.py -train ./data/train.json -valid ./data/valid.json | tee ./logs/train.log 2>&1

dt_end=$( date +"%s%3N" )
elapsed=$(( dt_end - dt_start ))
echo "実行時間:" ${elapsed} "[ms]"

# 実行時間: xx[ms]