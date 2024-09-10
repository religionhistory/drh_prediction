import pandas as pd
from evaluation_functions import multiple_lineplots, single_lineplot

# setup
outpath = "../figures/study3"

# first calculate squared difference for each case
pairwise_corr = pd.read_csv("evaluation/correlations_study3.csv")

pairwise_corr["corr_difference"] = (
    pairwise_corr["corr_complete"] - pairwise_corr["corr_imputed"]
)
pairwise_corr["corr_difference_abs"] = abs(pairwise_corr["corr_difference"])
pairwise_corr["corr_difference_sq"] = pairwise_corr["corr_difference"] ** 2
pairwise_corr["pairs"] = pairwise_corr["var_x"] + " " + pairwise_corr["var_y"]
# pairwise_corr_agg = pairwise_corr.rename(
#    columns={"corr_difference_sq": "Squared Difference"}
# )

# then aggregate
# pairwise_corr_agg = (
#    pairwise_corr_agg.groupby(["Type", "Percent", "Iter", "Method"])[
#        "Squared Difference"
#    ]
#    .mean()
#    .reset_index(name="Squared Difference")
# )

# I think the above is wrong, because we should first get the average deviation
# and then we should square that
mean_deviation = (
    pairwise_corr.groupby(["Type", "Percent", "Iter", "Method", "pairs"])
    .agg("corr_difference")
    .mean()
    .reset_index(name="mean deviation")
)
mean_deviation["squared deviation"] = mean_deviation["mean deviation"] ** 2

# single aggregated plot
single_lineplot(
    df=mean_deviation,
    metric="squared deviation",
    hue="Method",
    ncol_legend=3,
    outpath=outpath,
    outname="squared_deviation.png",
)

# by missingness mechanism
multiple_lineplots(
    df=mean_deviation,
    grid="Type",
    metric="squared deviation",
    hue="Method",
    sharey="all",
    ncol_legend=3,
    outpath=outpath,
    outname="squared_deviation_by_type.png",
)
