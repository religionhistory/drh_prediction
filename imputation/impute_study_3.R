source('imputation_functions.R')
library(missForest)
library(tidyverse)
print(getwd())

# log messages
options(warn = 1)
warning_file = file("output/study3/warnings.log", open = "wt")
sink(warning_file, type = "message")

# setup 
seed <- 658
study <- 3
m <- 5

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
print("Starting imputation for study: 3")

print('mice baseline')
mice_impute_baseline <- run_multiple_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = m, seed = seed)
write_imputations(data = mice_impute_baseline, study = study, algorithm = "mice")

print('MICE rf')
print(length(data_class))
mice_impute_rf <- run_multiple_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = m, seed = seed, method = 'rf')
write_imputations(data = mice_impute_rf, study = study, algorithm = "miceRF")

print('MICE pmm')
print(length(data_class))
mice_impute_pmm <- run_multiple_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = m, seed = seed, method = 'pmm')
write_imputations(data = mice_impute_pmm, study = study, algorithm = "micePMM")

print('MICE cart')
mice_impute_cart <- run_multiple_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = m, seed = seed, method = 'cart')
write_imputations(data = mice_impute_cart, study = study, algorithm = "miceCART")

print('MICE sample')
mice_impute_sample <- run_multiple_imputation(data_class, dataset_id_strings, question_level, mice_impute, m = m, seed = seed, method = 'sample')
write_imputations(data = mice_impute_sample, study = study, algorithm = "miceSample")

# close connection
sink(type = "message")