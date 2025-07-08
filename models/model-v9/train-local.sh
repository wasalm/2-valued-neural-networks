#!/bin/bash
cd "$(dirname "$0")"
source ./environment/bin/activate

# Use current timestamp for our run names
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP

# Make a folder for the current run
cd public
cd data
mkdir $TIMESTAMP

# Copy the configuration
cp ../../config.py $TIMESTAMP

# Create a directory listing for the websites
ls -d */ > folders.txt

# Run code
cd ../../
python train.py $TIMESTAMP 2>&1 | tee public/data/$TIMESTAMP/stdout.txt

