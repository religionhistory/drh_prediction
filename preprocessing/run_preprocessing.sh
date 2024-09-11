#!/bin/bash

# run each preprocessing step
python3 1_preprocess_answers.py
python3 2_preprocess_entries.py
python3 3_preprocess_datasets.py

echo "preprocessing done"
