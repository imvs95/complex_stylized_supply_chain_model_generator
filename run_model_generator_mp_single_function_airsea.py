"""
Created on: 6-8-2021 10:09

@author: IvS
"""
import time
import numpy as np
import copy

from complex_toy_model_graph_generator_airsea import ComplexSimModelGraph

from pydsol.core.experiment import SingleReplication
from pydsol.core.simulator import DEVSSimulatorFloat

import networkx as nx

from pydsol.model.basic_logger import get_module_logger
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(name)s - %(filename)s: line %(lineno)s)')

logger = get_module_logger(__name__, level=logging.INFO)


def run_simulation_single(rep, graph, interarrival_time=10, manufacturing_time=2.5, **kwargs):
    """
    This function automatically generates the simulation model with an user-defined Networkx Graph object. Other
    input parameters are the interarrival time of the products and the manufacturing time of the products.
    The function runs the simulation model with user-defined number of replications.

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
    replication, list, pd.DataFrame


    """
    # exp_output = {}
    # for rep in range(replications):

    # kwargs["lock"].acquire()
    #
    # try:
    graph_copy = copy.deepcopy(graph)
    del graph

    start_time = time.time()
    simulator = DEVSSimulatorFloat("sim" + str(rep))
    sim_model = ComplexSimModelGraph(simulator, graph=graph_copy, input_params={"interarrival_time": interarrival_time,
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
        time.sleep(0.01)

    if simulator.simulator_time == run_time:
        time.sleep(0.05)

    exp_output = sim_model.get_output_statistics()

    # convert to type that can handle memoryview
    exp_output_kpis = exp_output["outcomes"]
    kpis_array = [(key, value) for key, value in exp_output_kpis.items()]

    # exp_output_time_series = exp_output["time_series"]
    # column_names = exp_output_time_series.columns
    # # time_series_array = np.array(exp_output_time_series.values,
    # #                               dtype=[(name, exp_output_time_series[name].dtype) for name in column_names])
    # time_series_array = exp_output_time_series.to_numpy()

    del simulator
    del sim_model
    del replication
    del graph_copy

    # print("Replication {0} is finished in {1}".format(rep, time.time() - start_time))

    # time.sleep(np.random.uniform(0, 1))

    return rep + 1, kpis_array, exp_output["time_series"]

    # finally:
    #     kwargs["lock"].release()
