source('imputation/imputation_functions.R') # 'imputation_functions.R' 
library(missForest)
library(tidyverse)
print(getwd())

# setup 
seed <- 658
study <- 3 

# path to output folder
# modified_folder_path <- paste0("output/study3/missForest") # '../output/study3/missForest'
# if (!dir.exists(modified_folder_path)) {
#   dir.create(modified_folder_path)
# }

# use data from study 1 
folder_path <- paste0("imputation/output/study1/additional_NA") # '../output/study1/additional_NA'
file_names <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)

data_missing <- list()
for (file_path in file_names) {
  file_name <- basename(file_path)
  data <- read.csv(file_path)
  file_id <- sub(".*/NA_(.*)\\.csv", "\\1", file_path)
  data_missing[[file_id]] <- data
}

# Questions has the correct data types 
questions <- read_csv("data/preprocessed/answers.csv", show_col_types = FALSE) # '../data/preprocessed/answers.csv'
questions <- questions %>% select(question_id, data_type) %>% distinct()

# Create index of variable types for GLRM and conversion of data type
print('fix variable types')
var_types <- lapply(data_missing, variables_types)

# Convert variables to correct class
print('correct variable classes')
data_class <- Map(correct_class, data_missing, var_types)

## dataset id strings
question_level <- read_csv(paste0("data/preprocessed/question_level_study1.csv"), show_col_types = FALSE)  # '../data/preprocessed/question_level_study1.csv'
dataset_id_strings <- generate_datasets_ids_string(question_level, max_level = max(question_level$question_level))

mice_impute <- function(data, seed, m, ...) {
  entry_id_col <- data$entry_id 
  data <- data %>% 
    select(-entry_id)
  
  # fill constant columns
  data <- remove_na_from_constant(data)

  # run MICE imputation
  MICE_imp <- mice::mice(data, m = m, print = FALSE, seed = seed, nnet.MaxNWts = 10000, remove.collinear=FALSE, ...)

  # depending on m
  if (m == 1) {
    MICE_imputed <- mice::complete(MICE_imp)
    MICE_imputed <- cbind(entry_id_col, MICE_imputed)
    return(MICE_imputed)
  }
  else {
    # Initialize an empty list to store each imputed dataset
    imputed_datasets <- list()
    
    # Iterate through each imputation and bind the entry_id_col
    for(i in 1:m) {
      # Complete the dataset for the i-th imputation
      imputed_data <- mice::complete(MICE_imp, action = i)
      
      # Combine the entry_id_col with the imputed data
      imputed_data_with_id <- cbind(entry_id_col, imputed_data)
      
      # Store the combined data in the list
      imputed_datasets[[i]] <- imputed_data_with_id
    }
    
    return(imputed_datasets)
  }
} 


# implement this 
m=2
data_class_subset <- data_class[1:2]

# 
stage <- 1 
imputed_data_class <- duplicate_and_rename(data_class_subset, m) # we need m duplicates here 
columns_keep <- dataset_id_strings[[stage]]
data_class_subset <- subset_dataframes(data_class_list, columns_keep)
imp_subset <- lapply(data_class_subset, mice_impute_multiple, m=m, seed = seed)
imp_flattened <- unlist(imp_subset, recursive = FALSE)
for (i in seq_along(imputed_data_class)) {
  imputed_data_class[[i]][, columns_keep] <- imp_flattened[[i]][, columns_keep]
}
imputed_data_class <- lapply(imputed_data_class, function(df) update_child_entries(df, question_level, condition_fn_zero, 0))


run_multiple_imputation <- function(data_class, dataset_id_strings, question_level, imputation_function, m, ...) {
  
  # Initialize the imputed data with the input data
  imputed_data_class <- duplicate_and_rename(data_class, m)
  print('running imputation')
  
  for (stage in seq_along(dataset_id_strings)) {
    print(sprintf("imputation stage: %s", stage))
    
    # Identify the columns to keep for this stage
    columns_keep <- dataset_id_strings[[stage]]
    
    # Subset the original dataframes to keep only the relevant columns for this stage
    data_class_subset <- subset_dataframes(imputed_data_class, columns_keep)
    
    # Only multiple imputation for first stage 
    if (stage == 1){
      imp_subset <- lapply(data_class_subset, imputation_function, m=m, ...)
      imp_subset <- unlist(imp_subset, recursive = FALSE)
    } else {
      imp_subset <- lapply(data_class_subset, imputation_function, m=1, ...)
    }

    # Update the imputed_data_class with the new imputed values from forest_imp_subset
    for (i in seq_along(imputed_data_class)) {
      imputed_data_class[[i]][, columns_keep] <- imp_subset[[i]][, columns_keep]
    }

    # If parent question is no then child question should be no
    imputed_data_class <- lapply(imputed_data_class, function(df) update_child_entries(df, question_level, condition_fn_zero, 0))
  }

  return(imputed_data_class)
}

# random forest impute
# print('Random Forest imputation')
# rf_impute <- run_imputation(data_class, dataset_id_strings, question_level, rf_impute, seed = seed)
# write_imputations(data = rf_impute, study = study, algorithm = "missForest", appendix = "mf")

# mice impute
print('MICE imputation')
mice_impute_baseline <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, seed = seed)
write_imputations(data = mice_impute_baseline, study = study, algorithm = "mice", appendix = "mice")

# also why not run e.g., random forest
print('MICE rf')
mice_impute_rf <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, seed, method = 'rf')
write_imputations(data = mice_impute_rf, study = study, algorithm = "miceRF", appendix = "miceRF")

print('MICE pmm')
mice_impute_pmm <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, seed = seed, method = 'pmm')
write_imputations(data = mice_impute_pmm, study = study, algorithm = "micePMM", appendix = "micePMM")

print('MICE cart')
mice_impute_cart <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, seed = seed, method = 'cart')
write_imputations(data = mice_impute_cart, study = study, algorithm = "miceCART", appendix = "miceCART")

print('MICE sample')
mice_impute_sample <- run_imputation(data_class, dataset_id_strings, question_level, mice_impute, seed = seed, method = 'sample')
write_imputations(data = mice_impute_sample, study = study, algorithm = "miceSample", appendix = "miceSample")