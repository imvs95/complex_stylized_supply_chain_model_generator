"""
Created on: 6-8-2021 10:09

@author: IvS
"""
import time
from datetime import date
import pandas as pd
import numpy as np

from complex_toy_model_graph import ComplexSimModelGraph

from pydsol.core.experiment import SingleReplication, Experiment
from pydsol.core.simulator import DEVSSimulatorFloat

from utils.create_graph_xlsx import construct_graph, plot_graph
from utils.aggregate_statistics import aggregate_statistics
from utils.configure_input_data_xlsx import create_input_data_graph_from_excel

# from ema_workbench import Model, IntegerParameter, RealParameter, TimeSeriesOutcome, SequentialEvaluator

from pydsol.model.basic_logger import get_module_logger
import logging

logging.basicConfig(level=logging.CRITICAL,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(name)s - %(filename)s: line %(lineno)s)')

logger = get_module_logger(__name__, level=logging.INFO)


def run_simulation(graph, interarrival_time=1.5, manufacturing_time=2.5,
                   link_transit_import_port=9286, link_import_wholesales=135,
                   replications=10):
    """
    This function automatically generates the simulation model with an user-defined Networkx Graph object. Other
    input parameters are the interarrival time of the products, the manufacturing time of the products, the link
    transit import port and the link import wholesales. The function runs the simulation model with user-defined
    number of replications. It saves the timeseries and other KPI outcomes to a CSV file and in a dictionary.

    Parameters
    ----------
    graph: nx.Graph
        Networkx Graph object
    interarrival_time: Union[int, float]
        inputparameter for the interarrival time of the products. Default is 1.5 days.
    manufacturing_time: Union[int, float]
        inputparameter for the manufacturing time of the product. Default is 2.5 days.
    link_transit_import_port: int
        inputparameter for the link transit import port. Default is 9286.
    link_import_wholesales: int
        inputparameter for the link import wholesales. Default is 135.
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
                                                                               "manufacturing_time": manufacturing_time,
                                                                               "link_transit_import": link_transit_import_port,
                                                                               "link_import_wholesales": link_import_wholesales})
        sim_model.seed = rep
        run_time = 364  # 1 year
        replication = SingleReplication(str(rep), 0.0, 0.0, run_time)
        # experiment = Experiment("test", simulator, sim_model, 0.0, 0.0, 700, nr_replications=5)
        simulator.initialize(sim_model, replication)
        simulator.start()
        while simulator.simulator_time < run_time:
            time.sleep(0.01)

        if simulator.simulator_time == run_time:
            time.sleep(0.005)

        exp_output[rep + 1] = sim_model.get_output_statistics()


    logging.info("Experiment with {0} replications is finished".format(replications))

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
    df_avg_outcomes = pd.DataFrame.from_dict(avg_outcomes, orient="index")

    # # # Save as CSV
    date_today = date.today().strftime("%Y%m%d")
    all_timeseries.to_csv(
        "./data/" + date_today + "_" + str(sim_model.__class__.__name__) + "_GT_EventTimeSeries_ManufacturingTime" + str(
            round(manufacturing_time, 2)) + "_Runtime" + str(run_time) + ".csv")

    df_avg_outcomes.to_csv(
        "./data/" + date_today + "_" + str(sim_model.__class__.__name__) + "_GT_KPIs_ManufacturingTime" + str(
            round(manufacturing_time, 2)) + "_Runtime" + str(run_time) + ".csv")

    return {"time_series": all_timeseries, "avg_outcomes": avg_outcomes}


if __name__ == "__main__":
    logger.info("Start building the model")

    actors, distances = create_input_data_graph_from_excel(r"./input/graph_config_str.xlsx")

    logger.info("Input data is read")

    start_time = time.time()
    graph_supply_chain = construct_graph(actors, distances)
    # plot_graph(graph_supply_chain)

    logger.info("Graph is created from input data")

    reps = 1
    results = run_simulation(graph=graph_supply_chain, replications=reps)

    logger.info("Running the model for {0} replications takes {1:.1f} seconds".format(reps, (time.time() - start_time)))

    # for ground truth: make time series sparse before aggregating!

    aggr_time_series = aggregate_statistics(results["time_series"])

    kpis = results["avg_outcomes"]
