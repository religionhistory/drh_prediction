#!/bin/bash

# run each preprocessing step
Rscript extract_world_regions_1.R
python3 preprocess_answers_2.py
python3 preprocess_entries_3.py
python3 regions_coded_4.py
python3 preprocess_final_5.py

echo "preprocessing done"
