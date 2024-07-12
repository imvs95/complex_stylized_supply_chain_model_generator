"""
Created on: 6-8-2021 10:09

@author: IvS
"""
import time
from datetime import date
import pandas as pd
import numpy as np
import pickle

from pydsol.core.experiment import SingleReplication, Experiment
from pydsol.core.simulator import DEVSSimulatorFloat

from utils.create_graph_xlsx import construct_graph, plot_graph
from utils.aggregate_statistics import aggregate_statistics
from utils.configure_input_data_xlsx import create_input_data_graph_from_excel
from utils.configure_input_data_graph import add_input_params_to_db_structures

from structure_model_composer.sample import read_skeleton_data_base

# from ema_workbench import Model, IntegerParameter, RealParameter, TimeSeriesOutcome, SequentialEvaluator

from pydsol.model.basic_logger import get_module_logger
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(name)s - %(filename)s: line %(lineno)s)')

logger = get_module_logger(__name__, level=logging.INFO)

import multiprocessing as mp
from functools import partial

from run_model_generator_mp_single_function_airsea import run_simulation_single

# mp.log_to_stderr(logging.DEBUG)


def run_sim_model_replications(pool, graph, replications=10, **kwargs):
    """ Run the simulation model for a number of replications and aggregate the results using multiprocessing."""
    reps = list(range(replications))
    time.sleep(0.1)
    results_all = pool.map(partial(run_simulation_single, graph=graph, lock=kwargs["lock"]), reps)
    logging.info("Experiment with {0} replications is finished".format(replications))

    exp_output = {}
    for i in range(len(results_all)):
        exp_output[results_all[i][0]] = {"outcomes": {v[0]: v[1] for v in results_all[i][1:-1][0]},
                                         "time_series": pd.DataFrame(results_all[i][-1])}

    del graph
    del results_all

    all_timeseries_df = []
    # Add replication number
    for key in exp_output.keys():
        time_series = exp_output[key]["time_series"]
        time_series["Replications"] = key

        all_timeseries_df.append(time_series)

        outcomes = exp_output[key]["outcomes"]
        if key == next(iter(exp_output)):
            average_outcomes = {n: list() for n in outcomes.keys()}
        for name in outcomes.keys():
            kpi_value = outcomes[name]
            average_outcomes[name].append(kpi_value)

    # Combine all time series
    all_timeseries = pd.concat(all_timeseries_df).reset_index(drop=True)
    avg_outcomes = {k: np.mean(v) for k, v in average_outcomes.items()}
    # df_avg_outcomes = pd.DataFrame.from_dict(avg_outcomes, orient="index")

    # # # Save as CSV
    # date_today = date.today().strftime("%Y%m%d")
    # all_timeseries.to_csv(
    #     "./data/" + date_today + "_" + str(sim_model.__class__.__name__) + "_GT_EventTimeSeries_ManufacturingTime" + str(
    #         round(manufacturing_time, 2)) + "_Runtime" + str(run_time) + ".csv")
    #
    # df_avg_outcomes.to_csv(
    #     "./data/" + date_today + "_" + str(sim_model.__class__.__name__) + "_GT_KPIs_ManufacturingTime" + str(
    #         round(manufacturing_time, 2)) + "_Runtime" + str(run_time) + ".csv")

    del exp_output

    return {"time_series": all_timeseries, "avg_outcomes": avg_outcomes}





if __name__ == "__main__":
    logger.info("Start building the model")

    # db_structures = read_skeleton_data_base(r"./structure_model_composer/databases/location_model_isa_10")
    default_input_params = pd.read_excel(r"./input/default_input_params_actors.xlsx", sheet_name="actors")

    # new_db = add_input_params_to_db_structures(db_structures, default_input_params)

    logger.info("Input data is read")

    start_time = time.time()

    logger.info("Graph is loaded from input data")

    # graph_supply_chain = new_db.iloc[-1]["graph"]

    # for GT
    with open(r"./data/Ground_Truth_Graph_Topology_DF.pkl", "rb") as file:
        df_ground_truth = pickle.load(file)

    with open(r"./data/location_model_isa_5_cnhk_usa_betweenness.pkl", "rb") as file:
        hpc = pickle.load(file)

    # df_new = add_input_params_to_db_structures(df_ground_truth, default_input_params)
    # graph_supply_chain = df_ground_truth.iloc[0]["graph"]

    graph_supply_chain = hpc[1]["graph"]

    REPLICATIONS = 10

    lock = mp.Manager().Lock()

    with mp.Pool(processes=2) as pool:
        logger.info("Multiprocessing started")
        results = run_sim_model_replications(pool, graph=graph_supply_chain, replications=REPLICATIONS, lock=lock)

        logger.info("Running the model for {0} replications takes {1:.1f} seconds".format(REPLICATIONS, (time.time() - start_time)))

        # for ground truth: make time series sparse before aggregating!

        aggr_time_series = aggregate_statistics(results["time_series"])

        kpis = results["avg_outcomes"]
