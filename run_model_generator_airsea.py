"""
Created on: 6-8-2021 10:09

@author: IvS
"""
import time
from datetime import date
import pandas as pd
import numpy as np
import pickle
 
from complex_toy_model_graph_generator_airsea import ComplexSimModelGraph

from pydsol.core.experiment import SingleReplication, Experiment
from pydsol.core.simulator import DEVSSimulatorFloat

from utils.create_graph_xlsx import construct_graph, plot_graph
from utils.aggregate_statistics import aggregate_statistics
from utils.configure_input_data_xlsx import create_input_data_graph_from_excel
from utils.configure_input_data_graph_airsea import add_input_params_to_db_structures

from structure_model_composer.sample import read_skeleton_data_base
from structure_model_composer.set_locations_of_structures_airsea import plot_graph_locations_with_pos

# from ema_workbench import Model, IntegerParameter, RealParameter, TimeSeriesOutcome, SequentialEvaluator

from pydsol.model.basic_logger import get_module_logger
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(name)s - %(filename)s: line %(lineno)s)')

logger = get_module_logger(__name__, level=logging.INFO)


def run_simulation(graph, interarrival_time=10, manufacturing_time=1.5,
                   replications=10):
    """
    This function automatically generates the simulation model with an user-defined Networkx Graph object. Other
    input parameters are the interarrival time of the products and the manufacturing time of the products.
    The function runs the simulation model with user-defined number of replications.
    It saves the timeseries and other KPI outcomes to a CSV file and in a dictionary.

    Parameters
    ----------
    graph: nx.Graph
        Networkx Graph object
    interarrival_time: Union[int, float]
        inputparameter for the interarrival time of the products. Default is 1.5 days.
    manufacturing_time: Union[int, float]
        inputparameter for the manufacturing time of the product. Default is 2.5 days.
    replications: int
        number of replications. Default is 10.

    Returns
    -------
    dict


    """
    exp_output = {}
    for rep in range(replications):
        simulator = DEVSSimulatorFloat("sim")
        sim_model = ComplexSimModelGraph(simulator, graph=graph, input_params={"interarrival_time": interarrival_time,
                                                                               "manufacturing_time": manufacturing_time})
        sim_model.seed = rep
        simulator.seed = rep
        run_time = 364  # 2 year
        replication = SingleReplication(str(rep), 0.0, 0.0, run_time)
        # experiment = Experiment("test", simulator, sim_model, 0.0, 0.0, 700, nr_replications=5)
        simulator.initialize(sim_model, replication)
        simulator.start()
        # TODO Python wacht niet todat de simulatie voorbij is, vandaar deze while loop
        while simulator.simulator_time < run_time:
            time.sleep(0.04)

        if simulator.simulator_time == run_time:
            time.sleep(0.5)

        try:
            exp_output[rep + 1] = sim_model.get_output_statistics()
        except (RuntimeError, ValueError):
            time.sleep(10)
            exp_output[rep + 1] = sim_model.get_output_statistics()

        # exp_output[rep + 1] = sim_model.get_output_statistics()

        del simulator
        del sim_model
        del replication

    logging.info("Experiment with {0} replications is finished".format(replications))

    del graph

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

    # # # Save as CSV
    # df_avg_outcomes = pd.DataFrame.from_dict(avg_outcomes, orient="index")
    # date_today = date.today().strftime("%Y%m%d")
    # all_timeseries.to_csv(
    #     "./data/" + date_today + "_FINAL_" + str(sim_model.__class__.__name__) + "_GT_CNHK_USA_EventTimeSeries_ManufacturingTime" + str(
    #         round(manufacturing_time, 2)) + "_Runtime" + str(run_time) + ".csv")
    #
    # df_avg_outcomes.to_csv(
    #     "./data/" + date_today + "_FINAL_" + str(sim_model.__class__.__name__) + "_GT_CNHK_USA_KPIs_ManufacturingTime" + str(
    #         round(manufacturing_time, 2)) + "_Runtime" + str(run_time) + ".csv")

    return {"time_series": all_timeseries, "avg_outcomes": avg_outcomes}


if __name__ == "__main__":
    logger.info("Start building the model")

    # db_structures = read_skeleton_data_base(r"./data/location_test_airsea_graphs_10_betweenness")
    # default_input_params = pd.read_excel(r"./input/default_input_params_actors_airsea.xlsx", sheet_name="actors")
    #
    # new_db = add_input_params_to_db_structures(db_structures, default_input_params)


    logger.info("Input data is read")

    start_time = time.time()

    logger.info("Graph is loaded from input data")

    # # graph_supply_chain = new_db.iloc[-1]["graph"]
    # graph_supply_chain = new_db.iloc[48]["graph"]

    #for GT
    # with open(r"./data/Ground_Truth_Graph_Topology_DF_CNHK_USA.pkl", "rb") as file:
    #     df_ground_truth = pickle.load(file)

    with open(r"./data/location_model_hpc_10_cnhk_usa_airsea_betweenness.pkl", "rb") as file:
        hpc = pickle.load(file)

    # #for GT
    # df_new = add_input_params_to_db_structures(df_ground_truth, default_input_params)
    # graph_supply_chain = df_ground_truth.iloc[0]["graph"]

    graph_supply_chain = hpc[3]["graph"]

    reps = 1
    results = run_simulation(graph=graph_supply_chain, replications=reps)

    logger.info("Running the model for {0} replications takes {1:.1f} seconds".format(reps, (time.time() - start_time)))

    # for ground truth: make time series sparse before aggregating!

    aggr_time_series = aggregate_statistics(results["time_series"])

    kpis = results["avg_outcomes"]
