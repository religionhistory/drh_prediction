source('imputation_functions.R')
library(missForest)
library(tidyverse)

# log messages
options(warn = 1)
warning_file = file("output/warnings_temp.log", open = "wt")
sink(warning_file, type = "message")

# setup 
seed <- 658
study <- 1

# path to input folder 
folder_path <- paste0("output/study1/additional_NA")
file_names <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)

data_missing <- list()
for (file_path in file_names) {
  file_name <- basename(file_path)
  data <- read.csv(file_path)
  file_id <- sub(".*/NA_(.*)\\.csv", "\\1", file_path)
  data_missing[[file_id]] <- data
}

# Questions has the correct data types 
questions <- read_csv("../data/preprocessed/answers.csv", show_col_types = FALSE) 
questions <- questions %>% select(question_id, data_type) %>% distinct()

# Create index of variable types for GLRM and conversion of data type
var_types <- lapply(data_missing, variables_types)

# Convert variables to correct class
data_class <- Map(correct_class, data_missing, var_types)

# dataset id strings
question_level <- read_csv(paste0("../data/preprocessed/question_level_study1.csv"), show_col_types = FALSE)
dataset_id_strings <- generate_datasets_ids_string(question_level, max_level = max(question_level$question_level))

## start imputation ##
print(paste0("Starting imputation for study: ", 1))

print('Mode impute')
mode_impute <- run_imputation(data_class, dataset_id_strings, question_level, mode_impute)
write_imputations(data = mode_impute, study = study, algorithm = "mode")

print('missForest impute')
rf_impute <- run_imputation(data_class, dataset_id_strings, question_level, rf_impute, seed = seed)
write_imputations(data = rf_impute, study = study, algorithm = "missForest")

print('MICE baseline')
mice_impute_baseline <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = 1, seed = seed)
write_imputations(data = mice_impute_baseline, study = study, algorithm = "mice")

print('MICE rf')
mice_impute_rf <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = 1, seed = seed, method = 'rf')
write_imputations(data = mice_impute_rf, study = study, algorithm = "miceRF")

print('MICE pmm')
mice_impute_pmm <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = 1, seed = seed, method = 'pmm')
write_imputations(data = mice_impute_pmm, study = study, algorithm = "micePMM")

print('MICE cart')
mice_impute_cart <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = 1, seed = seed, method = 'cart')
write_imputations(data = mice_impute_cart, study = study, algorithm = "miceCART")

print('MICE sample')
mice_impute_sample <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = 1, seed = seed, method = 'sample')
write_imputations(data = mice_impute_sample, study = study, algorithm = "miceSample")

# close connection
sink(type = "message")