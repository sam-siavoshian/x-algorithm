#!/bin/bash
# one heavy job at a time: control run, then the launch cache, then the embedding
until [ -s data/expand/runs/ctl_s1/metrics.json ] || ! pgrep -f "runs/ctl_s1" > /dev/null; do sleep 3; done
pkill -CONT -f score_launch.py
until ! pgrep -f "score_launch.py" > /dev/null; do sleep 3; done
pkill -CONT -f embed_rows.py
