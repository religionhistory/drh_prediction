def remove_all_nan(group_cols, metrics_combined):
    metrics_combined = metrics_combined.copy()
    metrics_combined["is_nan"] = metrics_combined["value"].isna()
    nan_groups = metrics_combined.groupby(group_cols)["is_nan"].all().reset_index()
    nan_combinations = nan_groups[nan_groups["is_nan"] == True][group_cols]
    df_clean = metrics_combined.merge(
        nan_combinations, on=group_cols, how="left", indicator=True
    )
    df_clean = df_clean[df_clean["_merge"] == "left_only"].drop(
        columns=["is_nan", "_merge"]
    )
    return df_clean


def fill_zero(metrics_combined):
    df_fillzero = metrics_combined.copy()
    df_fillzero["value"] = df_fillzero["value"].fillna(0.0)
    return df_fillzero


def remove_any_nan(group_cols, metrics_combined):
    df_removenan = metrics_combined.copy()
    nan_any_combination = df_removenan[df_removenan["value"].isna()][
        group_cols
    ].drop_duplicates()
    df_removenan = df_removenan.merge(
        nan_any_combination, on=group_cols, how="left", indicator=True
    )
    df_removenan = df_removenan[df_removenan["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    return df_removenan


import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def multiple_lineplots(
    df,
    metric,
    hue,
    grid,
    color_dict=None,  # Allows custom color for each hue value
    label_dict=None,  # Allows custom label for each metric
    legend=True,
    legend_order=None,  # Added legend_order parameter
    figsize=(7, 7),
    alpha=1,
    errorbar=("ci", 95),
    sharey="none",
    ncol_legend=None,
    outpath=None,
    outname=None,
    y_tick_decimals=None,  # New parameter
):
    unique_grid_values = df[grid].unique()
    n_subplots = len(unique_grid_values)
    fig, ax = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    fig.patch.set_facecolor("white")

    # Ensure ax is always iterable
    axes = np.atleast_1d(ax)

    # Initialize variables to determine global y-axis limits if needed
    global_ymin = float("inf")
    global_ymax = float("-inf")

    for num, ele in enumerate(unique_grid_values):
        # Select the subset of data for the current grid value
        df_subset = df[df[grid] == ele]

        # Use the provided legend order or default to unique hue values
        hue_values = legend_order if legend_order else df_subset[hue].unique()

        for hue_value in hue_values:
            df_grouped = df_subset[df_subset[hue] == hue_value]

            # Plotting with specified color from the color_dict or default color
            plot_color = (
                color_dict.get(hue_value)
                if color_dict and hue_value in color_dict
                else None
            )

            sns.lineplot(
                data=df_grouped,
                x="missing_percent",
                y="value",
                label=hue_value if legend else None,
                color=plot_color,
                alpha=alpha,
                marker="o",
                ax=axes[num],
                errorbar=errorbar,
            )

            # Update global y-axis limits
            # ymin, ymax = df_grouped["average"].min(), df_grouped["average"].max()
            ymin, ymax = df_grouped["value"].min(), df_grouped["value"].max()
            global_ymin, global_ymax = min(global_ymin, ymin), max(global_ymax, ymax)

        # Remove the legend from individual plots
        ele = label_dict[ele] if label_dict and ele in label_dict else ele
        axes[num].legend().remove()
        axes[num].set_ylabel("")
        axes[num].set_title(f"{ele}", fontsize=12)
        axes[num].set_xlabel("Percent Missing", fontsize=12)
        x_ticks = df_subset["missing_percent"].unique()
        axes[num].set_xticks(x_ticks)

    # If sharey='all', set the same y-axis limits for all subplots
    global_ymin = global_ymin - global_ymax * 0.05
    global_ymax = global_ymax + global_ymax * 0.05
    if sharey == "all":
        for a in axes:
            a.set_ylim(global_ymin, global_ymax)

    # Set y-axis tick formatter to control decimal places
    if y_tick_decimals is not None:
        formatter = FuncFormatter(lambda y, _: f"{y:.{y_tick_decimals}f}")
        for a in axes:
            a.yaxis.set_major_formatter(formatter)

    # Set a global y-axis label for the entire figure instead of each subplot
    fig.text(
        0.05, 0.5, "Average " + metric, va="center", rotation="vertical", fontsize=12
    )

    # Only add the legend if specified
    plt.tight_layout()
    if legend:
        handles, labels = axes[0].get_legend_handles_labels()

        # Reorder the legend handles and labels according to the provided legend_order
        if legend_order:
            # Create a dictionary mapping labels to handles
            handle_dict = dict(zip(labels, handles))
            handles = [
                handle_dict[label] for label in legend_order if label in handle_dict
            ]
            labels = [label for label in legend_order if label in handle_dict]

        ncol = ncol_legend if ncol_legend else len(handles)
        nrow = int(np.ceil(n_subplots / ncol))
        y_align_legend = (-0.1 * nrow) + 0.05
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, y_align_legend),
            ncol=ncol,
            frameon=False,
            fontsize=12,
        )

    plt.subplots_adjust(left=0.15, bottom=0.1)

    # Save or show
    if outpath:
        plt.savefig(os.path.join(outpath, outname), dpi=300, bbox_inches="tight")
    else:
        plt.show()
