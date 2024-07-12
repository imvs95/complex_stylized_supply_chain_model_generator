"""
Created on: 26-6-2023 11:31

@author: IvS
"""
import pickle
import datetime
import sys
import numpy as np

def remove_redudant_info_graph(graph, keys_edges_to_keep, keys_nodes_to_keep):
    """This function removes the redundant information from a graph.

    Parameters:
        graph (networkx.Graph): Graph to remove redundant information from
        keys_edges_to_keep (list): List of keys to keep in the edges
        keys_nodes_to_keep (list): List of keys to keep in the nodes

    Returns:
        graph (networkx.Graph): Graph with redundant information removed
        """
    for (o, d, attr) in graph.edges(data=True):
        if isinstance(o, (str, np.str_)) and isinstance(d, (str, np.str_)):
            attr_to_remove_edge = [key for key in attr if key not in keys_edges_to_keep]
            for key in attr_to_remove_edge:
                graph.edges[(o, d)].pop(key)
        else:
            continue

    for (n, attr) in graph.nodes(data=True):
        if isinstance(n, (str, np.str_)):
            attributes_to_remove = [key for key in attr if key not in keys_nodes_to_keep]
            for key in attributes_to_remove:
                graph.nodes[n].pop(key)
        else:
            continue
    return graph


if __name__ == "__main__":
    db_name_pkl = "location_model_hpc_40000_betweenness"
    db_location = "random_hpc_40000_2"

    with open(r"../data/{0}.pkl".format(db_name_pkl), "rb") as file:
        large_dict = pickle.load(file)

    keys_edges = set(["frequency", "median_time_minutes", "distribution"])
    keys_nodes = set(["latitude", "longitude", "processing_distribution_full", "processing_distribution",
                              "location", "node_type", "entity_type", "pos",
                      'capacity', 'transfer_in_time', 'processing_time', 'distribution', 'vehicle_type'])
    mem = 0
    for k, attr in large_dict.items():
        new_graph = remove_redudant_info_graph(attr["graph"], keys_edges, keys_nodes)
        large_dict[k]["graph"] = new_graph

        graph_mem = np.sum([sys.getsizeof(e) for e in new_graph.edges]) + np.sum([sys.getsizeof(n) for n in new_graph.nodes])
        mem += graph_mem

    print(f"{datetime.datetime.now():%Y-%m-%d}" + " Size of graph is {0} MB".format(mem/1024/1024))

    #  Save new
    network_topology = "betweenness"

    with open(r"../data/{0}_{1}.pkl".format(db_location, network_topology), "wb") as file:
        pickle.dump(large_dict, file)

    print(f"{datetime.datetime.now():%Y-%m-%d}" + "Graphs are made smaller and saved to pickle".format(network_topology))
