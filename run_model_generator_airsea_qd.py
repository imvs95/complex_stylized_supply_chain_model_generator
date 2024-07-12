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
                   wholesales_consolidator_time=(0.5, 0.5, 1), wholesales_consolidator_pickuptime=(0.5, 1, 2),
                   wholesales_consolidator_prob_on_container=0.5, export_port_sea_time=(1, 2, 2),
                   transit_port_sea_time=(1, 2, 2), import_wait_on_steck_time_sea=(0.5, 3),
                   import_prob_extracting_sea=0.5,
                   export_port_air_time=(0.5, 1, 1), transit_port_air_time=(0.5, 1, 1),
                   import_wait_on_steck_time_air=(0.5, 1), import_prob_extracting_air=0.5,
                   wholesales_distributor_time=(0.5, 1, 2), large_retailer_time=0.2, small_retailer_time=0.1, replications=10):
    """
    This function automatically generates the simulation model with an user-defined Networkx Graph object. Other input
    parameters are actor specific. The function runs the simulation model with user-defined
    number of replications. It saves the timeseries and other KPI outcomes to a CSV file and in a dictionary.

    Parameters
    ----------
    graph: nx.Graph
        Networkx Graph object
    interarrival_time: Union[int, float]
    manufacturing_time: Union[int, float]
    wholesales_consolidator_time: Tuple[int, int, int]
    wholesales_consolidator_pickuptime: Tuple[int, int, int]
    wholesales_consolidator_prob_on_container: float
    export_port_sea_time: Tuple[int, int, int]
    transit_port_sea_time: Tuple[int, int, int]
    import_wait_on_steck_time_sea: Tuple[int, int]
    import_prob_extracting_sea: float
    export_port_air_time: Tuple[int, int, int]
    transit_port_air_time: Tuple[int, int, int]
    import_wait_on_steck_time_air: Tuple[int, int]
    import_prob_extracting_air: float
    wholesales_distributor_time: Tuple[int, int, int]
    large_retailer_time: float
    small_retailer_time: float
    replications: int
        number of replications. Default is 10.

    Returns
    -------
    dict



    """
    exp_output = {}
    # all_reps = list(range(replications))
    for rep in range(replications):
        simulator = DEVSSimulatorFloat("sim")
        sim_model = ComplexSimModelGraph(simulator, graph=graph, input_params={"interarrival_time": interarrival_time,
                                                                               "manufacturing_time": manufacturing_time,
                                                                               "wholesales_consolidator_time": wholesales_consolidator_time,
                                                                               "wholesales_consolidator_pickuptime": wholesales_consolidator_pickuptime,
                                                                               "wholesales_consolidator_prob_on_container": wholesales_consolidator_prob_on_container,
                                                                               "export_port_sea_time": export_port_sea_time,
                                                                               "transit_port_sea_time": transit_port_sea_time,
                                                                               "import_wait_on_steck_time_sea": import_wait_on_steck_time_sea,
                                                                               "import_prob_extracting_sea": import_prob_extracting_sea,
                                                                               "export_port_air_time": export_port_air_time,
                                                                               "transit_port_air_time": transit_port_air_time,
                                                                               "import_wait_on_steck_time_air": import_wait_on_steck_time_air,
                                                                               "import_prob_extracting_air": import_prob_extracting_air,
                                                                               "wholesales_distributor_time": wholesales_distributor_time,
                                                                               "large_retailer_time": large_retailer_time,
                                                                               "small_retailer_time": small_retailer_time})
        sim_model.seed = rep
        simulator.seed = rep
        run_time = 364  # 1 year
        replication = SingleReplication(str(rep), 0.0, 0.0, run_time)
        # experiment = Experiment("test", simulator, sim_model, 0.0, 0.0, 700, nr_replications=5)
        simulator.initialize(sim_model, replication)
        simulator.start()
        # TODO Python wacht niet todat de simulatie voorbij is, vandaar deze while loop
        while simulator.simulator_time < run_time:
            time.sleep(0.01)

        if simulator.simulator_time == run_time:
            time.sleep(0.1)

        try:
            exp_output[rep + 1] = sim_model.get_output_statistics()
            #exp_output[f"{rep}_{n}"] = sim_model.get_output_statistics()
        except RuntimeError:
            time.sleep(2)
            exp_output[rep + 1] = sim_model.get_output_statistics()
            #exp_output[f"{rep}_{n}"] = sim_model.get_output_statistics()

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

    del exp_output

    # # # Save as CSV
    # df_avg_outcomes = pd.DataFrame.from_dict(avg_outcomes, orient="index")
    # date_today = date.today().strftime("%Y%m%d")
    # all_timeseries.to_csv(
    #     "./data/" + date_today + "_QD_" + str(sim_model.__class__.__name__) + "_GT_CNHK_USA_EventTimeSeries_ManufacturingTime" + str(
    #         round(manufacturing_time, 2)) + "_Runtime" + str(run_time) + ".csv")
    #
    # df_avg_outcomes.to_csv(
    #     "./data/" + date_today + "_QD_" + str(sim_model.__class__.__name__) + "_GT_CNHK_USA_KPIs_ManufacturingTime" + str(
    #         round(manufacturing_time, 2)) + "_Runtime" + str(run_time) + ".csv")
    return {"time_series": all_timeseries, "avg_outcomes": avg_outcomes}


if __name__ == "__main__":
    logger.info("Start building the model")

    start_time = time.time()

    # with open(r"./data/location_new_graphs_10_betweenness.pkl", "rb") as file:
    #     hpc = pickle.load(file)

    logger.info("Graph is loaded from input data")

    # # graph_supply_chain = new_db.iloc[-1]["graph"]
    # graph_supply_chain = new_db.iloc[48]["graph"]

    # #for GT
    with open(r"./data/Ground_Truth_Graph_Topology_DF_CNHK_USA.pkl", "rb") as file:
        df_ground_truth = pickle.load(file)

    # db_structures = read_skeleton_data_base(r"./data/location_test_airsea_graphs_10_betweenness")
    default_input_params = pd.read_excel(r"./input/default_input_params_actors_airsea_qd.xlsx", sheet_name="actors")
    #default_input_params = pd.read_excel(r"./input/default_input_params_actors_airsea.xlsx", sheet_name="actors")
    #
    # hpc = add_input_params_to_db_structures(pd.DataFrame.from_dict(hpc, orient="index"), default_input_params).to_dict(orient="index")

    logger.info("Input data is read")

    # #for GT
    df_new = add_input_params_to_db_structures(df_ground_truth, default_input_params)
    graph_supply_chain = df_ground_truth.iloc[0]["graph"].copy()

    # graph_supply_chain = hpc[10]["graph"]

    reps = 10
    results = run_simulation(graph=graph_supply_chain, replications=reps)

    logger.info("Running the model for {0} replications takes {1:.1f} seconds".format(reps, (time.time() - start_time)))

    # for ground truth: make time series sparse before aggregating!

    aggr_time_series = aggregate_statistics(results["time_series"])

    kpis = results["avg_outcomes"]

    # graph_supply_chain = df_ground_truth.iloc[0]["graph"].copy()
    # results_2 = run_simulation(graph=graph_supply_chain, replications=reps)
    # a = aggregate_statistics(results_2["time_series"])
