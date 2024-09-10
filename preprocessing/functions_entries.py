# mode feature
def mode_feature_by_entry(df, entry_column, feature_columns):
    # Ensure feature_columns is a list, even if it's a single column name
    if not isinstance(feature_columns, list):
        feature_columns = [feature_columns]

    # Group by entry_column and feature_columns and count occurrences
    group_columns = [entry_column] + feature_columns
    df_features = df.groupby(group_columns).size().reset_index(name="count")

    # Sort by entry_column and count, then drop duplicates based on entry_column
    sorted_df = df_features.sort_values(
        by=[entry_column, "count"], ascending=[True, False]
    )
    dedup_df = sorted_df.drop_duplicates(subset=entry_column)
    dedup_df = dedup_df.drop(columns=["count"])

    # Count the number of unique combinations for each entry_column
    unique_count_df = df_features.groupby(entry_column).size().reset_index(name="count")
    unique_count_df["unique"] = unique_count_df["count"] == 1
    unique_count_df = unique_count_df.drop(columns=["count"])

    # merge with dedup_df
    final_df = dedup_df.merge(unique_count_df, on=entry_column)

    # Testing
    assert df[entry_column].nunique() == len(final_df)
    return final_df
