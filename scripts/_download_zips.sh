#!/bin/bash
set -e
D=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG
cd $D/data
rm -rf ptbxl ptbxl_plus
mkdir -p zip && cd zip
nohup curl -sS -L --retry 10 --retry-delay 30 -o ptbxl.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip > $D/logs/ptbxl_zip.log 2>&1 &
echo PID_PTBXL=$!
nohup curl -sS -L --retry 10 --retry-delay 30 -o ptbxlplus.zip https://physionet.org/static/published-projects/ptb-xl-plus/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip > $D/logs/ptbxlplus_zip.log 2>&1 &
echo PID_PLUS=$!
sleep 10
ls -la .
