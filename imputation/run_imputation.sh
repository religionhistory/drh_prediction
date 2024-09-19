#!/bin/bash

# run each imputation stage
python s1_add_na.py
echo "s1_add_na.py done"
python s1_tune.py
echo "s1_tune.py done"
python s1_impute.py
echo "s1_impute.py done"
python s2_add_na.py
echo "s2_add_na.py done"
python s2_tune.py
echo "s2_tune.py done"
python s2_impute.py
echo "s2_impute.py done"

echo "imputation done for all studies"
