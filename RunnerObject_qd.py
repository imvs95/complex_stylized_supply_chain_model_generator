"""
Created on: 9-2-2022 16:19

@author: IvS
"""
import pandas as pd
import numpy as np
import time

from run_model_generator_airsea_qd import run_simulation
from utils.aggregate_statistics import aggregate_statistics


class ComplexSupplyChainSimModelQD(object):
    """ This class is a wrapper for the simulation model. It contains a method to run the simulation model automatically
     from a graph and to calculate the statistics of the simulation model. This class is used for the
     quality diversity algorithm."""
    @staticmethod
    def run(parameters: list):
        start_time = time.time()
        reps = 10
        # # this is with the distance of the min to mode, and mode to max
        results = run_simulation(graph=parameters[0], interarrival_time=parameters[1], manufacturing_time=parameters[2],
                                 wholesales_consolidator_time=(parameters[3], parameters[3]+parameters[4],
                                                               parameters[3]+parameters[4]+parameters[5]),
                                 wholesales_consolidator_pickuptime=(parameters[6], parameters[6]+parameters[7],
                                                                     parameters[6]+parameters[7]+parameters[8]),
                                 wholesales_consolidator_prob_on_container=parameters[9],
                                 export_port_sea_time=(parameters[10], parameters[10]+parameters[11],
                                                       parameters[10]+parameters[11]+parameters[12]),
                                 transit_port_sea_time=(parameters[13], parameters[13]+parameters[14],
                                                        parameters[13]+parameters[14]+parameters[15]),
                                 import_wait_on_steck_time_sea=(parameters[16], parameters[16]+parameters[17]),
                                 import_prob_extracting_sea=parameters[18],
                                 export_port_air_time=(parameters[19], parameters[19]+parameters[20],
                                                       parameters[19]+parameters[20]+parameters[21]),
                                 transit_port_air_time=(parameters[22], parameters[22]+parameters[23],
                                                        parameters[22]+parameters[23]+parameters[24]),
                                 import_wait_on_steck_time_air=(parameters[25], parameters[25]+parameters[26]),
                                 import_prob_extracting_air=parameters[27],
                                 wholesales_distributor_time=(parameters[28], parameters[28]+parameters[29],
                                                              parameters[28]+parameters[29]+parameters[30]),
                                 large_retailer_time=parameters[31],
                                 small_retailer_time=parameters[32], replications=reps)

        # results = run_simulation(graph=parameters[0], interarrival_time=parameters[1], manufacturing_time=parameters[2],
        #                          wholesales_consolidator_time=(parameters[3], parameters[4], parameters[5]),
        #                          wholesales_consolidator_pickuptime=(parameters[6], parameters[7], parameters[8]),
        #                          wholesales_consolidator_prob_on_container=parameters[9],
        #                          export_port_sea_time=(parameters[10], parameters[11], parameters[12]),
        #                          transit_port_sea_time=(parameters[13], parameters[14], parameters[15]),
        #                          import_wait_on_steck_time_sea=(parameters[16], parameters[17]),
        #                          import_prob_extracting_sea=parameters[18],
        #                          export_port_air_time=(parameters[19], parameters[20], parameters[21]),
        #                          transit_port_air_time=(parameters[22], parameters[23], parameters[24]),
        #                          import_wait_on_steck_time_air=(parameters[25], parameters[26]),
        #                          import_prob_extracting_air=parameters[27],
        #                          wholesales_distributor_time=(parameters[28], parameters[29], parameters[30]),
        #                          large_retailer_time=parameters[31],
        #                          small_retailer_time=parameters[32], replications=reps)

        #print("Experiment with {0} replications is finished in {1:.2f}".format(reps, time.time() - start_time))

        del parameters

        aggr_results = aggregate_statistics(results["time_series"])

        kpis = results["avg_outcomes"]

        del results

        return aggr_results, kpis
