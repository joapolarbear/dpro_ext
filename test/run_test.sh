#!/bin/bash
# Please make sure navigating to the project's root directory before running the script.

CURPATH=$(realpath .)
export PYTHONPATH=$CURPATH:$PYTHONPATH

python3 $@