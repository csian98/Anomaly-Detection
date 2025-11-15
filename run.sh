#/bin/bash

TARGET="$1"

if [ -z "$TARGET" ]; then
	echo "Usage: $0 <SCRIPT>"
	exit 1
fi
   
LOG_FILE=`date +"%Y%m%d%H%M%S"`

EXT="${TARGET##*.}"

if [ "$EXT" = "py" ]; then
	nohup python ./$TARGET > ./tmp/log/$LOG_FILE 2>&1 &
else
	nohup ./$TARGET > ./tmp/log/$LOG_FILE 2>&1 &
fi

