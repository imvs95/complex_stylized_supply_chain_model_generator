"""
Created on: 9-2-2022 16:19

@author: IvS
"""
import pandas as pd
import numpy as np

from run_model import run_simulation


class ComplexSupplyChainSimModel(object):
    """ This class is a wrapper for the simulation model. It contains a method to run the simulation model and to
    calculate the statistics of the simulation model. """
    @staticmethod
    def run(parameters: list):
        results = run_simulation(interarrival_time=parameters[0], manufacturing_time=parameters[1],
                                 link_transit_import_port=parameters[2], link_import_wholesales=parameters[3],
                                 replications=20)
        aggr_results = results["time_series"].groupby(["Time"]).mean()
        time_series = aggr_results.iloc[:, :-1]

        statistics = ComplexSupplyChainSimModel.calculate_statistics(time_series)

        kpis = results["avg_outcomes"]

        return statistics, kpis

    @staticmethod
    def calculate_statistics(df_in: pd.DataFrame):

        df_out = pd.DataFrame(index=["mean", "std", "p5", "p95", "avg_interval_t"], columns=df_in.columns)
        df_out.loc["mean"] = [np.mean(df_in[col]) for col in df_in.columns]
        df_out.loc["std"] = [np.std(df_in[col]) for col in df_in.columns]
        df_out.loc["p5"] = [np.nanpercentile(df_in[col], 5) for col in df_in.columns]
        df_out.loc["p95"] = [np.nanpercentile(df_in[col], 95) for col in df_in.columns]
        df_out.loc["avg_interval_t"] = [ComplexSupplyChainSimModel.calculate_average_interval_time(df_in[col]) for col in df_in.columns]

        return df_out

    @staticmethod
    def calculate_average_interval_time(column):
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
