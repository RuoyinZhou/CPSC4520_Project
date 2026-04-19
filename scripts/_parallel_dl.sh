#!/bin/bash
# Parallel ranged download of PhysioNet zip, to bypass per-connection throttling.
# Usage: _parallel_dl.sh URL OUTPUT N_PARALLEL
URL=$1; OUT=$2; N=${3:-16}
SIZE=$(curl -sLI "$URL" | awk 'tolower($1)=="content-length:" {gsub(/\r/,""); s=$2} END{print s}')
if [ -z "$SIZE" ] || [ "$SIZE" = "0" ]; then echo "FAIL: no content-length"; exit 1; fi
echo "size=$SIZE parallel=$N"
PART_DIR=${OUT}.parts && mkdir -p $PART_DIR
CHUNK=$(( SIZE / N ))
rm -f $PART_DIR/*
for i in $(seq 0 $((N-1))); do
  LO=$((i * CHUNK))
  HI=$(( (i+1) * CHUNK - 1 ))
  if [ $i = $((N-1)) ]; then HI=$((SIZE - 1)); fi
  curl -sS -L --retry 10 --retry-delay 5 -r ${LO}-${HI} -o $PART_DIR/part_$(printf %03d $i) "$URL" &
done
wait
cat $PART_DIR/part_* > $OUT && rm -rf $PART_DIR
ls -la $OUT
