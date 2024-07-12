"""
Created on: 16-1-2023 11:09

@author: IvS
"""
from typing import List

import networkx as nx
import matplotlib.pyplot as plt


from utils.configure_input_data_xlsx import create_input_data_graph_from_excel

def construct_nodes(graph, n_actors: [dict], distances: [dict]):
    """ Construct nodes for the graph based on the input data.
    Parameters:
        graph (nx.DiGraph): graph to which nodes are added
        n_actors (dict): dictionary with the actors and their attributes
        distances (dict): dictionary with the distances between the actors """
    sequence_actors: List[str] = list(n_actors)

    for idx, (name, attributes) in enumerate(n_actors.items()):
        distance_links = {k: v for (k, v) in distances.items() if name in k}
        ingoing_links = {k: v for (k, v) in distance_links.items() if name == k[1]}
        outgoing_links = {k: v for (k, v) in distance_links.items() if name == k[0]}

        if name == "wholesales_distributor":
            new_ingoing = [[i[0] for i in list(ingoing_links.values())[0]], [i[1] for i in list(ingoing_links.values())[0]]]
            ingoing_links[list(ingoing_links.keys())[0]] = new_ingoing

        elif name == "large_retailer":
            new_ingoing = [[i[0] for i in list(ingoing_links.values())[0]], [i[1] for i in list(ingoing_links.values())[0]],
                           [i[2] for i in list(ingoing_links.values())[0]]]
            ingoing_links[list(ingoing_links.keys())[0]] = new_ingoing

        elif name == "small_retailer":
            new_ingoing = [[i[0] for i in list(ingoing_links.values())[0]], [i[1] for i in list(ingoing_links.values())[0]],
                           [i[2] for i in list(ingoing_links.values())[0]], [i[3] for i in list(ingoing_links.values())[0]]]
            ingoing_links[list(ingoing_links.keys())[0]] = new_ingoing

        for n in range(attributes["num"]):
            dict_attributes = dict(idx=idx,
                                   n=n,
                                   node_type=name,
                                   entity_type=name,
                                   pos=(idx, n),
                                   #location=attributes["location"][n], make x,y coordinates of location and calculate distance links accordingly
                                   ingoing_entity=sequence_actors[sequence_actors.index(name) - 1] if idx != 0 else None,
                                   outgoing_entity=sequence_actors[sequence_actors.index(name) + 1] if idx != len(
                                       sequence_actors) - 1 else None,
                                   ingoing_link_dist=[[v[n]] if attributes["num"] > 1 else v for k, v in ingoing_links.items()][0] if idx != 0 else None,
                                   outgoing_link_dist=[[v[n]] if attributes["num"] > 1 else v for k, v in outgoing_links.items()][0] if idx != len(
                                       sequence_actors) - 1 else None)

            if name == "import_port":
                dict_attributes["outgoing_link_dist"] = dict_attributes["outgoing_link_dist"][0]
            elif name == "wholesales_distributor":
                dict_attributes["ingoing_link_dist"] = dict_attributes["ingoing_link_dist"][0]
                dict_attributes["outgoing_link_dist"] = dict_attributes["outgoing_link_dist"][0]
            elif name == "large_retailer":
                dict_attributes["ingoing_link_dist"] = dict_attributes["ingoing_link_dist"][0]
                dict_attributes["outgoing_link_dist"] = dict_attributes["outgoing_link_dist"][0]
            elif name == "small_retailer":
                dict_attributes["ingoing_link_dist"] = dict_attributes["ingoing_link_dist"][0]

            #update dict with attributes of input data
            try:
                additional_attributes = {k: v[n] for k, v in attributes.items() if k != "num"}
            except IndexError:
                additional_attributes = {k: v[n] for k, v in attributes.items() if k != "num" and len(v) == attributes["num"]}

            dict_attributes.update(additional_attributes)


            #add node
            graph.add_node(str(name)+"_"+str(n), **dict_attributes)

def construct_edges(graph):
    """ Construct edges for the graph based on the input data."""
    for node_r, attributes_r in graph.nodes(data=True):
        node_list = [(node_r, node, {"distance_km": [item for item in attributes_r["outgoing_link_dist"] for item2 in attributes["ingoing_link_dist"] if item == item2][0]}) for node, attributes in graph.nodes(data=True)
                     if attributes_r["outgoing_entity"] == attributes["node_type"]]
        graph.add_edges_from(node_list)


def construct_graph(actors, distances):
    """ Construct a graph based on the input data.

    Parameters:
        actors (dict): dictionary with the actors and their attributes
        distances (dict): dictionary with the distances between the actors

    Returns:
        G (nx.DiGraph): graph with nodes and edges based on the input data"""
    G = nx.DiGraph()
    construct_nodes(G, actors, distances)
    construct_edges(G)

    return G

#Plot graph
def plot_graph(graph):
    """ Plot the graph."""
    pos = {k: v["pos"] for k, v in graph.nodes(data=True)}
    nx.draw_networkx(graph, pos, with_labels=True, font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "length"), font_size=8)
    plt.show()

if __name__ == "__main__":
    actors, distances = create_input_data_graph_from_excel(r"../input/graph_config_str.xlsx")
    graph_supply_chain = construct_graph(actors, distances)
    plot_graph(graph_supply_chain)


