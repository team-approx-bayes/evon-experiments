#!/bin/bash

# change to the directory of the script
cd "$(dirname "$0")"

PATH_TO_DATASET="datasets/c4-t5/subset/"

echo "Downloading the dataset for pretraining the LLM..."
# Download the dataset for pretraining the LLM. This script assumes that you have the necessary permissions to access the dataset
uv run download_dataset.py --save-dir $PATH_TO_DATASET
echo "Dataset downloaded successfully. The dataset is saved in $PATH_TO_DATASET/"

echo "Linking the dataset to the current directory..."
# link the dataset to the current directory if not in this directory
if [ ! -d "./datasets/c4-t5/subset" ]; then
    ln -s $PATH_TO_DATASET ./datasets/c4-t5/subset
fi
echo "Dataset linked successfully. You can now access the dataset in the current directory under ./datasets"