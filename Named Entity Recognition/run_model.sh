#!/bin/sh
if [ "$1" = "train" ]; then
    python3 train.py "$2" "$3"
fi

if [ "$1" = "test" ]; then 
    python3 test.py "$2" "$3"
fi

