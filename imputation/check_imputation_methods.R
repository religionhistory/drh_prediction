source("imputation/imputation_functions.R")
library(missForest)
library(tidyverse)
library(mice)
print(getwd())

remove_na_from_constant <- function(df) {
  relevant_cols <- names(df)[startsWith(names(df), "X")]

  for (col_name in relevant_cols) {
    # Since we now assume all these columns are factors, directly proceed to check for single level
    if (length(levels(df[[col_name]])) == 1) {
      # Replace NAs with the observed level
      observed_level <- levels(df[[col_name]])[1]
      df[[col_name]][is.na(df[[col_name]])] <- observed_level
    }
  }

  return(df)
}

mice_impute <- function(data, seed, m, ...) {
  entry_id_col <- data$entry_id
  data <- data %>%
    select(-entry_id)

  # fill constant columns
  data <- remove_na_from_constant(data)

  # run MICE imputation
  MICE_imp <- mice::mice(data, m = m, print = FALSE, seed = seed, maxit = 200, nnet.MaxNWts = 10000, remove.collinear = FALSE, ...)

  # depending on m
  if (m == 1) {
    MICE_imputed <- mice::complete(MICE_imp)
    MICE_imputed <- cbind(entry_id_col, MICE_imputed)
    return(MICE_imputed)
  } else {
    # Initialize an empty list to store each imputed dataset
    imputed_datasets <- list()

    # Iterate through each imputation and bind the entry_id_col
    for (i in 1:m) {
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

# setup
study <- 1

# quick test
data <- data.frame(X1 = c(1, 0, 0, NA, 1), X2 = c(NA, 0, 1, 0, 1), X3 = c(NA, 0, 1, 0, 1))
mice_imp <- mice::mice(data, seed = 658, m = 1, remove_collinear = FALSE)
summary(mice_imp)
mice_imp <- mice::complete(mice_imp)
mice_imp

# path to input folder
folder_path <- paste0("imputation/output/study", study, "/additional_NA")
file_names <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)

data_missing <- list()
for (file_path in file_names) {
  file_name <- basename(file_path)
  data <- read.csv(file_path)
  file_id <- sub(".*/NA_(.*)\\.csv", "\\1", file_path)
  data_missing[[file_id]] <- data
}

# Questions has the correct data types
questions <- read_csv("data/preprocessed/answers.csv", show_col_types = FALSE)
questions <- questions %>%
  select(question_id, data_type) %>%
  distinct()

# Create index of variable types for GLRM and conversion of data type
print("fix variable types")
var_types <- lapply(data_missing, variables_types)

# Convert variables to correct class
print("correct variable classes")
data_class <- Map(correct_class, data_missing, var_types)

## dataset id strings
question_level <- read_csv(paste0("data/preprocessed/question_level_study", study, ".csv"), show_col_types = FALSE)
dataset_id_strings <- generate_datasets_ids_string(question_level, max_level = max(question_level$question_level))

# okay now just do it for 1 to test:
imputed_data_class <- data_class
stage <- 1
columns_keep <- dataset_id_strings[[stage]]
data_class_subset <- subset_dataframes(imputed_data_class, columns_keep)

# just 1 case;
m <- 1
seed <- 658
data_case <- data_class_subset[[3]]
data_case
entry_id_col <- data_case$entry_id
data_case <- data_case %>%
  select(-entry_id)

# fill constant columns
data_case <- remove_na_from_constant(data_case)

# run MICE imputation
# here if we set another seed it works ...?
MICE_imp <- mice::mice(data_case, m = m, print = FALSE, seed = seed, nnet.MaxNWts = 10000, remove.collinear = FALSE)
summary(MICE_imp) # logged events are collinear (X5137 is the main problem, with X5132 and maybe X5196)
head(MICE_imp$loggedEvents)
MICE_imputed <- mice::complete(MICE_imp)

# what if we remove the variable that is problematic? #
data_case_remove <- data_case %>%
  select(-X5137)
mice_imp_remove <- mice::mice(data_case_remove, m = m, print = FALSE, seed = seed, nnet.MaxNWts = 10000, remove.collinear = FALSE) # yep, works now.

# this also does the trick
data_case_rem2 <- data_case %>%
  select(-X5132)
mice_imp_remove <- mice::mice(data_case_rem2, m = m, print = FALSE, seed = seed, nnet.MaxNWts = 10000, remove.collinear = FALSE) # yep, works now.

# try only selecting those two instead
# but this also works...?
data_case_rem3 <- data_case %>%
  select(-X4654, -X4676, -X4729, -X4745, -X4776, -X4780, -X4794, -X4827, -X4954, -X5196, -X5220, -X5226)
data_case_rem3
mice_imp_rem3 <- mice::mice(data_case_rem3, m = m, print = FALSE, seed = seed, nnet.MaxNWts = 10000, remove.collinear = FALSE) # yep, works now.
data_case_rem3

# save this to look at in python (they do not seem perfectly collinear)
write.csv(data_case, "temporary.csv", row.names = FALSE)

##### see above for the convergence issue #####

# imputation methods
forest_imp_subset <- lapply(data_class_subset, function(x) rf_impute(data = x, seed = 658))
mice_imp_subset <- lapply(data_class_subset, function(x) mice_impute(data = x, m = 1, seed = 658))
mice_imp_subset

# what if we wanted to do the simplest possible
# just impute the most common value for each column

# we want to see the warnings though
mice_test <- function(data, seed) {
  entry_id_col <- data$entry_id
  data <- data %>%
    select(-entry_id)

  # fill constant columns
  data <- remove_na_from_constant(data)

  MICE_imp <- mice::mice(data, m = 1, print = FALSE, seed = seed, nnet.MaxNWts = 10000, remove.collinear = FALSE)
  return(MICE_imp)
}
names(data_class_subset)

data_case <- data_class_subset[[5]]
data_case <- remove_na_from_constant(data_case)
mice_imp <- mice_test(data_case, 512)
head(mice_imp$loggedEvents) # mostly just logging logistic regression and constants
d <- mice::complete(mice_imp)
head(d)
length(levels(data_case[["X4699"]]))
d[["X4699"]]
d[["X4701"]]
levels(d[["X4699"]])
data_case[["X4699"]] <- as.factor(data_case[["X4699"]])
questions %>% filter(question_id == 4699)
column_types <- sapply(data_case, class)
column_types
## check mode impute ##
mode_imp_subset <- lapply(data_class_subset, function(x) mode_impute(x, seed))
mode_imp_subset

# 4699: what is it?
imputed_data_class <- data_class
print("running imputation")
for (stage in seq_along(dataset_id_strings)) {
  print(sprintf("imputation stage: %s", stage))
  # Identify the columns to keep for this stage
  columns_keep <- dataset_id_strings[[stage]]

  # Subset the original dataframes to keep only the relevant columns for this stage
  data_class_subset <- subset_dataframes(imputed_data_class, columns_keep)

  # Impute missing values using the subset data
  # Make sure that the imputation function returns a list of dataframes with the same structure as data_class_subset
  forest_imp_subset <- lapply(data_class_subset, function(x) rf_impute(data = x, seed = seed))

  # Update the imputed_data_class with the new imputed values from forest_imp_subset
  for (i in seq_along(imputed_data_class)) {
    # Make sure that forest_imp_subset has the same structure and order of rows as imputed_data_class
    # Replace the columns with the newly imputed data
    imputed_data_class[[i]][, columns_keep] <- forest_imp_subset[[i]][, columns_keep]
  }

  # If parent question is no then child question should be no
  imputed_data_class <- lapply(imputed_data_class, function(df) update_child_entries(df, question_level, function(x) x == 0, 0))
}
