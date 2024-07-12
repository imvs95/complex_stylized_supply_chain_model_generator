"""
Created on: 9-2-2022 16:19

@author: IvS
"""
import pandas as pd
import numpy as np
import multiprocessing as mp

from run_model_generator_mp import run_sim_model_replications
from utils.aggregate_statistics import aggregate_statistics


class ComplexSupplyChainSimModel(object):
    """ This class is a wrapper for the simulation model. It contains a method to run the simulation model automatically
     from a graph and to calculate the statistics of the simulation model. This class is used in the multiprocessing."""
    @staticmethod
    def run(parameters: list, pool_obj):
        lock = mp.Manager().Lock()

        results = run_sim_model_replications(pool_obj, graph=parameters[0], replications=5, lock=lock)

        del parameters
        del lock

        aggr_results = aggregate_statistics(results["time_series"])

        kpis = results["avg_outcomes"]

        del results

        return aggr_results, kpis