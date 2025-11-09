#/bin/bash

LOG_FILE=`date +"%Y%m%d%H%M%S"`
nohup python ./main.py > ./tmp/log/$LOG_FILE 2>&1 &
