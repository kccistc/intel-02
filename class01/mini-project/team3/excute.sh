#!/bin/bash

python3 401ObjectDetection.py
if [ $? -eq 0 ]; then
    echo "success to excute first script"
    python main.py
else
    echo "fail to excute first script"
fi
