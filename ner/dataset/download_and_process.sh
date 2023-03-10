#!/bin/bash

set -e

# Configure download location
DOWNLOAD_PATH="$BIOBERT_DATA"
if [ "$BIOBERT_DATA" == "" ]; then
    echo "BIOBERT_DATA not set; downloading to default path ('data')."
    DOWNLOAD_PATH="./data"
fi
DOWNLOAD_PATH_TAR="$DOWNLOAD_PATH.tar.gz"

# Download datasets
wget http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/datasets.tar.gz -O "$DOWNLOAD_PATH_TAR"
tar -xvzf "$DOWNLOAD_PATH_TAR"
rm "$DOWNLOAD_PATH_TAR"
mkdir src
mv datasets/NER/* ./src
rm -r datasets

echo "BioBERT dataset download done!"

# process
echo "Start processing"
python ./scripts/process.py

MAX_LENGTH=128