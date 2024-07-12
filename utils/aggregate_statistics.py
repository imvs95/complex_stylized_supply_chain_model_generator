"""
Created on: 10-2-2023 11:22

@author: IvS
"""
import pandas as pd
import numpy as np
from functools import reduce


def aggregate_statistics(df_timeseries):
    """ Aggregate the statistics of the time series for each actor in the supply chain. This is done based on
    the location of the actor in the supply chain. The statistics are calculated for each replication and then
    averaged over all replications. We focus on the quantity of the actor in the supply chain."""
    aggr_timeseries = pd.DataFrame()

    for rep in df_timeseries["Replications"].unique():
        df_rep = df_timeseries[df_timeseries["Replications"] == rep]

        # create aggregated statistic from each type of actor in the supply chain
        # change to "location" ipv "type" to get it for each location
        dict_rep = {}
        for i in df_rep['type'].unique():
            loc_df = df_rep[df_rep["type"] == i]
            dict_rep[i] = {"location": loc_df["location"].unique(), "df": loc_df[["time", "quantity"]]}

            for k, v in dict_rep.items():
                dict_rep[k]["df"] = v["df"].rename(columns={"quantity": k}).groupby(["time"]).sum()

        dfs = [v["df"] for k, v in dict_rep.items()]
        combined_dfs = reduce(lambda left, right: pd.merge(left, right, on="time",
                                                           how='outer'), dfs).fillna(0).sort_index()

        # add replication and combine with others
        combined_dfs["Replications"] = rep
        aggr_timeseries = pd.concat([aggr_timeseries, combined_dfs])

    # get mean of all replications
    mean_time_series = aggr_timeseries.groupby(["time"]).mean()
    final_time_series = mean_time_series.drop(columns=['Replications'])

    # calculate statistics of the time series
    statistics = calculate_statistics(final_time_series)

    return statistics


def calculate_statistics(df_in: pd.DataFrame):
    """Calculate statistics of the time series."""
    df_out = pd.DataFrame(index=["mean", "std", "p5", "p95", "avg_interval_t"], columns=df_in.columns)
    df_out.loc["mean"] = [np.mean(df_in[col]) for col in df_in.columns]
    df_out.loc["std"] = [np.std(df_in[col]) for col in df_in.columns]
    df_out.loc["p5"] = [np.nanpercentile(df_in[col], 5) for col in df_in.columns]
    df_out.loc["p95"] = [np.nanpercentile(df_in[col], 95) for col in df_in.columns]
    df_out.loc["avg_interval_t"] = [calculate_average_interval_time(df_in[col]) for col in df_in.columns]

    return df_out


def calculate_average_interval_time(column):
    """Calculate average interval time of an actor in the supply chain based on time series."""
    shifted_column = column.shift(1)
    # if previous item on list is smaller than current
    bool_comparison = shifted_column < column
    # get times
    index_restock = bool_comparison.index[bool_comparison == True]

    if len(index_restock) == 0:
        avg_interval_time = 0

    else:
        # determine interval times
        interval_times = [index_restock[0] - 0] + [index_restock[n] - index_restock[n - 1] for n in
                                                   range(1, len(index_restock))]

        # calculate average
        avg_interval_time = np.mean(interval_times)

    return avg_interval_time