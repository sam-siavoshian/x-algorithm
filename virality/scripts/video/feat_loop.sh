#!/bin/bash
# run the thumbnail feature extractor in chunks until every fetched thumbnail has features and the fetcher has stopped
while true; do
  nice -n 15 .venv/bin/python scripts/video/thumb_features.py 1500 2>&1 | grep -E "to do|FEATS_DONE"
  n=$(ls data/video/thumbs | wc -l); m=$(.venv/bin/python -c "import polars as pl;print(pl.read_parquet('data/video/thumb_feats.parquet').height)")
  if [ "$m" -ge "$n" ] && ! pgrep -f fetch_thumbs >/dev/null; then echo FEATLOOP_DONE; break; fi
  [ "$m" -ge "$n" ] && sleep 60
done
