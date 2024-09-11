library(tidyverse)
library(stringr)

# Create index of variable types for GLRM and conversion of data type
variables_types <- function(data) {
  var_types <- data.frame(ID = colnames(data)) %>%
    mutate(question_id = gsub("_.*$", "", ID)) %>%
    mutate(question_id = gsub("[^0-9.]", "", question_id)) %>% # [^0-9.-]
    mutate(question_id = as.numeric(question_id)) %>%
    left_join(questions, by = "question_id") %>%
    mutate(loss_func = case_when(
      ID == "entry_id" ~ "Categorical",
      ID %in% c("year_from", "year_to", "log_area") ~ "Absolute",
      str_detect(ID, "^region_") ~ "Categorical", # Match any ID starting with "region_"
      data_type %in% c("Nominal", "Nominal - Other", "Nominal - Multiple") ~ "Categorical",
      data_type %in% c("Discrete", "Continuous") ~ "Absolute"
    ))
}

# Convert variables to correct class
correct_class <- function(data, var_types) {
  continuous <- filter(var_types, data_type == "Continuous")
  discrete <- filter(var_types, data_type == "Discrete")
  nominal <- filter(var_types, data_type == "Nominal" | data_type == "Nominal - Multiple" | data_type == "Nominal - Other")
  data <- data %>%
    mutate(entry_id = factor(entry_id)) %>%
    mutate(across(year_from:year_to, as.integer)) %>%
    mutate(log_area = as.numeric(log_area)) %>%
    mutate(across(region_africa:region_southwest_asia, factor)) %>%
    mutate(across(nominal$ID, factor)) %>%
    mutate(across(continuous$ID, as.numeric)) %>%
    mutate(across(discrete$ID, as.integer))
  data <- as.data.frame(data)
}


# RF impute
rf_impute <- function(data, seed) {
  # Save entry_id column and remove it from the data
  entry_id_col <- data$entry_id
  data_without_entry_id <- data %>%
    select(-entry_id)

  # Perform imputation on data without entry_id
  forest_imp <- missForest(data_without_entry_id, variablewise = TRUE)
  forest_df <- forest_imp$ximp

  # Add the entry_id column back to the imputed data
  forest_df <- cbind(entry_id = entry_id_col, forest_df)

  return(forest_df)
}

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
  MICE_imp <- mice::mice(data, m = m, print = FALSE, seed = seed, nnet.MaxNWts = 10000, remove.collinear = FALSE, ...)

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

mode_impute <- function(data) {
  # Function to calculate the mode for a factor column
  calculate_mode <- function(x) {
    mode_level <- names(sort(table(x), decreasing = TRUE))[1]
    return(mode_level)
  }

  # Identify columns that start with "X"
  relevant_cols <- names(data)[startsWith(names(data), "X")]

  # Apply mode imputation to each relevant column
  for (col_name in relevant_cols) {
    # Calculate mode for the column
    mode_value <- calculate_mode(data[[col_name]])

    # Replace NA values with the mode
    data[[col_name]][is.na(data[[col_name]])] <- mode_value
  }

  return(data)
}

# Save files
write_imputations <- function(data, study, algorithm) {
  # Create the base directory based on the study number
  base_dir <- paste0("output/study", study)

  # Create the algorithm-specific directory if it doesn't exist
  algo_dir <- file.path(base_dir, algorithm)
  if (!dir.exists(algo_dir)) {
    dir.create(algo_dir, recursive = TRUE)
  }

  # Use mapply to write each element of the data list to its own CSV file
  invisible(mapply(write_csv, data, paste0(algo_dir, "/", algorithm, "_", names(data), ".csv"), SIMPLIFY = FALSE))
}

# helper function for update child entries
# when we want to check for zero
condition_fn_zero <- function(values) {
  # Return FALSE where values are NA, and check equality to zero otherwise
  return(!is.na(values) & values == 0)
}

update_child_entries <- function(df_wide, df_relationships, condition_fn, action_value) {
  # Filter for question_level > 1 and then sort to respect hierarchy
  df_relationships <- df_relationships %>%
    filter(question_level > 1) %>%
    arrange(question_level)

  # Prepare df_relationships for processing
  df_relationships <- df_relationships %>%
    mutate(
      child_col = paste0("X", question_id),
      parent_col = paste0("X", parent_question_id)
    )

  # Iterate over each relationship to update df_wide based on NA in parent
  for (i in 1:nrow(df_relationships)) {
    row <- df_relationships[i, ]
    child_col <- row$child_col
    parent_col <- row$parent_col

    # Identify rows where parent is NA
    na_rows <- condition_fn(df_wide[[parent_col]])

    # Propagate NA to child in those rows
    df_wide[na_rows, child_col] <- action_value
  }

  return(df_wide)
}

find_question_level <- function(question_id, drh) {
  computed_levels <- list() # Use a list for caching in R

  inner_find_question_level <- function(question_id) {
    # Base case: if the parent question ID is 0, the level is 0
    if (question_id == 0) {
      return(0)
    }
    # If already computed, return the stored level
    if (!is.null(computed_levels[[as.character(question_id)]])) {
      return(computed_levels[[as.character(question_id)]])
    }

    # Recursive case: find the parent question's ID and level
    parent_id <- drh[drh$question_id == question_id, ]$parent_question_id
    level <- inner_find_question_level(parent_id) + 1
    # Store the computed level in the cache
    computed_levels[[as.character(question_id)]] <- level
    return(level)
  }

  return(inner_find_question_level(question_id))
}


generate_datasets_ids_string <- function(df, max_level) {
  datasets_ids <- list() # Use a list to store IDs strings keyed by level

  # For level 1, simply take the question IDs
  # Ensure we always return a data frame with drop = FALSE
  question_ids_level_1 <- df[df$question_level == 1, "question_id", drop = FALSE]
  # Apply paste0 to each element of the column
  datasets_ids[[1]] <- paste0("X", question_ids_level_1$question_id)

  encountered_parents <- numeric(0) # Initialize an empty numeric vector for encountered parents
  for (i in 2:max_level) {
    level_i_questions <- df[df$question_level == i, "question_id", drop = FALSE]

    # Find the parents of level i questions
    parents_of_level_i <- unique(df[df$question_id %in% level_i_questions$question_id, "parent_question_id", drop = FALSE])

    # Add to list of encountered parents
    encountered_parents <- unique(c(encountered_parents, parents_of_level_i$parent_question_id))

    # For the second and subsequent datasets, include question IDs not in encountered_parents and up to current level
    # Ensure we always return a data frame with drop = FALSE
    question_ids_level_i <- df[!df$question_id %in% encountered_parents & df$question_level <= i, "question_id", drop = FALSE]
    # Apply paste0 to each element of the column
    datasets_ids[[i]] <- paste0("X", question_ids_level_i$question_id)
  }

  return(datasets_ids)
}


subset_dataframes <- function(dataframes_list, columns_to_keep) {
  # Function to subset a single dataframe based on the criteria
  subset_dataframe <- function(df) {
    # Identify columns that do not start with "X"
    non_x_columns <- names(df)[!grepl("^X", names(df))]

    # From the columns that start with "X", keep those that are in columns_to_keep
    x_columns_in_df <- intersect(names(df)[grepl("^X", names(df))], columns_to_keep)

    # Combine the non-X columns with the allowed X columns and subset the dataframe
    columns_to_subset <- c(non_x_columns, x_columns_in_df)
    return(df[, columns_to_subset, drop = FALSE])
  }

  # Apply the subsetting function to each dataframe in the list
  modified_dataframes_list <- lapply(dataframes_list, subset_dataframe)

  return(modified_dataframes_list)
}


# Define the main function for imputation
run_imputation <- function(data_class, dataset_id_strings, question_level, imputation_function, ...) {
  # Initialize the imputed data with the input data
  imputed_data_class <- data_class
  print("running imputation")

  for (stage in seq_along(dataset_id_strings)) {
    print(sprintf("imputation stage: %s", stage))

    # Identify the columns to keep for this stage
    columns_keep <- dataset_id_strings[[stage]]

    # Subset the original dataframes to keep only the relevant columns for this stage
    data_class_subset <- subset_dataframes(imputed_data_class, columns_keep)

    # Impute missing values using the subset data
    # The imputation function is now a parameter and can be either rf_impute, mode_impute, or any other specified
    # imp_subset <- lapply(data_class_subset, function(x) imputation_function(data = x, ...))
    imp_subset <- lapply(data_class_subset, imputation_function, ...)

    # Update the imputed_data_class with the new imputed values from forest_imp_subset
    for (i in seq_along(imputed_data_class)) {
      imputed_data_class[[i]][, columns_keep] <- imp_subset[[i]][, columns_keep]
    }

    # If parent question is no then child question should be no
    imputed_data_class <- lapply(imputed_data_class, function(df) update_child_entries(df, question_level, condition_fn_zero, 0))
  }

  return(imputed_data_class)
}


#### for multiple imputation ####
# Function to duplicate each element in the list n times
# It also renames the elements according to the pattern you provided
duplicate_and_rename <- function(list_of_dataframes, n) {
  # Initialize an empty list to store the results
  duplicated_list <- list()

  # Loop over each element in the list
  for (name in names(list_of_dataframes)) {
    for (i in 1:n) {
      # Create the new name by appending the duplicate number
      new_name <- paste(name, i, sep = "_")

      # Add the dataframe to the new list with the new name
      duplicated_list[[new_name]] <- list_of_dataframes[[name]]
    }
  }

  return(duplicated_list)
}


run_multiple_imputation <- function(data_class, dataset_id_strings, question_level, imputation_function, m, ...) {
  # Initialize the imputed data with the input data
  imputed_data_class <- duplicate_and_rename(data_class, m)
  print("running imputation")

  for (stage in seq_along(dataset_id_strings)) {
    print(sprintf("imputation stage: %s", stage))

    # Identify the columns to keep for this stage
    columns_keep <- dataset_id_strings[[stage]]

    # Subset the original dataframes to keep only the relevant columns for this stage
    data_class_subset <- subset_dataframes(imputed_data_class, columns_keep)

    # Only multiple imputation for first stage
    if (stage == 1) {
      imp_subset <- lapply(data_class_subset, imputation_function, m = m, ...)
      imp_subset <- unlist(imp_subset, recursive = FALSE)
    } else {
      imp_subset <- lapply(data_class_subset, imputation_function, m = 1, ...)
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
