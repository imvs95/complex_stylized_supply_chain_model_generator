"""
Created on: 8-12-2023 09:41

@author: IvS
"""
import pickle
import networkx as nx


def remove_edges_to_self(graph):
    """ Removes edges that have the same origin and destination"""
    self_loops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(self_loops)
    return graph


# if __name__ == "__main__":
#     with open(r"../input/graph_networkx_CNHK_USA_FINAL_route_with_sea.pkl", 'rb') as f:
#         G = pickle.load(f)
#
#     # Preprocessing Graph
#     G = remove_edges_to_self(G)
