"""
Created on: 9-2-2022 16:19

@author: IvS
"""
import pandas as pd
import numpy as np

# from run_model_generator import run_simulation
from run_model_generator_airsea import run_simulation
from utils.aggregate_statistics import aggregate_statistics


class ComplexSupplyChainSimModel(object):
    """ This class is a wrapper for the simulation model. It contains a method to run the simulation model automatically
     from a graph and to calculate the statistics of the simulation model. """
    @staticmethod
    def run(parameters: list):
        results = run_simulation(graph=parameters[0], replications=5)
                                 #interarrival_time=parameters[0], manufacturing_time=parameters[1],

        del parameters

        aggr_results = aggregate_statistics(results["time_series"])

        kpis = results["avg_outcomes"]

        del results

        return aggr_results, kpis