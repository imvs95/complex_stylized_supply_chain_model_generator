"""
Created on: 26-6-2023 11:31

@author: IvS
"""
import pickle
import datetime
import sys
import numpy as np

def remove_redudant_info_modality(graph):
    """This function removes the redundant information from a graph for the multiple of other modality.
    We only keep the unique values."""

    for (o, d, attr) in graph.edges(data=True):
        if isinstance(o, (str, np.str_)) and isinstance(d, (str, np.str_)):
            if "other_modality" in attr:
                    mod = graph.edges[(o, d)]["other_modality"]
                    graph.edges[(o, d)]["other_modality"] = list(set(mod))
            else:
                continue
        else:
            continue

    for (n, attr) in graph.nodes(data=True):
        if isinstance(n, (str, np.str_)):
            if "other_modality" in attr:
                mod = graph.edges[n]["other_modality"]
                graph.nodes[n]["other_modality"] = list(set(mod))
            else:
                continue
        else:
            continue
    return graph


if __name__ == "__main__":
    db_name_pkl = "location_model_hpc_50000_cnhk_usa_airsea"
    db_location = "location_model_hpc_40000_cnhk_usa_airsea"

    with open(r"../data/{0}.pkl".format(db_name_pkl), "rb") as file:
        large_dict = pickle.load(file)

    # mem = 0
    # for k, attr in large_dict.items():
    #     new_graph = remove_redudant_info_modality(attr["graph"])
    #     large_dict[k]["graph"] = new_graph
    #
    #     graph_mem = np.sum([sys.getsizeof(e) for e in new_graph.edges]) + np.sum([sys.getsizeof(n) for n in new_graph.nodes])
    #     mem += graph_mem
    #
    # print(f"{datetime.datetime.now():%Y-%m-%d}" + " Size of graph is {0} MB".format(mem/1024/1024))

    # #  Save new
    # network_topology = "betweenness"
    #
    # with open(r"../data/{0}_{1}.pkl".format(db_location, network_topology), "wb") as file:
    #     pickle.dump(large_dict, file)

    # new_dict = {k:v for k, v in large_dict.items() if v["index"] <= 10}
    graph_with_error = [(k, v) for k, v in large_dict.items() if
                        sum(1 for node in v["graph"].nodes() if v["graph"].out_degree(node) == 0) > 1]

    for i in graph_with_error:
        large_dict.pop(i[0])

    total_g = 40000
    total_error = len([g for g in graph_with_error if g[1]["index"] <= total_g])

    new_dict = {k: v for k, v in large_dict.items() if v["hash"] == "ground_truth" or v["index"] <= (total_g+total_error)}


    sorted_dict = dict(sorted(new_dict.items()))
    new_dict = {i: value for i, (key, value) in enumerate(sorted_dict.items())}

    with open(r"../data/{0}.pkl".format(db_location), "wb") as file:
        pickle.dump(new_dict, file)

    #print(f"{datetime.datetime.now():%Y-%m-%d}" + "Graphs are made smaller and saved to pickle".format(network_topology))
