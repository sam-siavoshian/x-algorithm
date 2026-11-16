#!/bin/bash
# resume the paused embedding as soon as the control run has written its metrics
until [ -s data/expand/runs/ctl_s1/metrics.json ] || ! pgrep -f "runs/ctl_s1" > /dev/null; do sleep 5; done
pkill -CONT -f embed_rows.py
