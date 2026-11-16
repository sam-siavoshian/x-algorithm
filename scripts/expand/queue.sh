#!/bin/bash
# Runs the job files in data/expand/queue in name order, one at a time (these machines thrash with two heavy jobs).
# A job is a shell script NNN_name.sh; when it ends a .done (or .failed) marker is written. New jobs can be added while it runs.
Q=data/expand/queue
while true; do
  next=""
  for f in $(ls $Q/*.sh 2>/dev/null | sort); do
    if [ ! -e "${f%.sh}.done" ] && [ ! -e "${f%.sh}.failed" ]; then next=$f; break; fi
  done
  if [ -z "$next" ]; then [ -e $Q/STOP ] && exit 0; sleep 5; continue; fi
  echo "$(date +%T) start $next" >> $Q/log.txt
  if bash "$next" > "${next%.sh}.out" 2>&1; then touch "${next%.sh}.done"; else touch "${next%.sh}.failed"; fi
  echo "$(date +%T) end $next $(ls ${next%.sh}.done ${next%.sh}.failed 2>/dev/null)" >> $Q/log.txt
done
