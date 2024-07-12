"""
Created on: 27-6-2023 14:14

@author: IvS
"""
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from utils.configure_input_data_xlsx import create_input_data_graph_from_excel
from utils.configure_input_data_graph import add_input_params_to_db_structures

from utils.create_graph_xlsx import construct_graph

from structure_model_composer.set_locations_of_structures_airsea import combine_shippings_routes_to_subgraph, \
    calculate_network_topology_and_add_to_df, plot_graph_locations_with_pos, remove_redudant_info_graph

def merge_graph_ports_with_generated_structure(generated_structure, graph_ports, mapping_locations):
    """ Merge the generated structure with the graph_ports"""

    graph_no_ports = generated_structure.subgraph([n for n, d in generated_structure.nodes(data=True)
                              if "port" not in d["node_type"]])
    all_graphs = nx.compose_all([graph_no_ports, graph_ports])
    combined_graph = nx.relabel_nodes(all_graphs, mapping_locations)

    #Find edges that are in the generated structure but not in the combined graph
    edges_to_add = [(l, r, d) for l, r, d in generated_structure.edges(data=True) if
             (l, r) not in combined_graph.edges() and l in combined_graph.nodes()
             and r in combined_graph.nodes()]

    combined_graph.add_edges_from(edges_to_add)

    combined_graph_name = nx.relabel_nodes(combined_graph, {v: k for k, v in mapping_locations.items()})

    #plot graph
    node_positions = nx.get_node_attributes(combined_graph_name, 'pos')
    nx.draw(combined_graph_name, with_labels=True, pos=node_positions)
    plt.show()

    return combined_graph_name

if __name__ == "__main__":
    actors, distances = create_input_data_graph_from_excel(r"../input/graph_config_str_groundtruth_cnhkusa_airsea.xlsx")
    graph_supply_chain = construct_graph(actors, distances)

    # manual changes for GT model (xlsx not working propely with distances in last part)
    graph_supply_chain.edges["import_port_1", "wholesales_distributor_1"]["distance_km"] = 72
    graph_supply_chain.edges["import_port_3", "wholesales_distributor_1"]["distance_km"] = 80
    graph_supply_chain.edges["wholesales_distributor_0", "large_retailer_2"]["distance_km"] = 0
    graph_supply_chain.edges["wholesales_distributor_1", "large_retailer_2"]["distance_km"] = 150
    graph_supply_chain.edges["large_retailer_1", "small_retailer_1"]["distance_km"] = 140
    graph_supply_chain.edges["large_retailer_1", "small_retailer_2"]["distance_km"] = 60
    graph_supply_chain.edges["large_retailer_2", "small_retailer_3"]["distance_km"] = 50

    # remove zero distance edges except for end customer
    edges_to_remove = [edge for edge in graph_supply_chain.edges(data=True) if
                       edge[2]["distance_km"] == 0 and "end_customer" not in edge[1]]
    graph_supply_chain.remove_edges_from(edges_to_remove)

    with open(r"../input/graph_networkx_CNHK_USA_FINAL_route_with_sea.pkl", 'rb') as f:
        G = pickle.load(f)


    np.random.seed(1)
    #only for GT
    # flight via hotspot AMS
    all_paths_hkg_bos = list(nx.all_simple_edge_paths(G, source="HKG", target="BOS", cutoff=2))
    path_hkg_ams_lax = [f for f in all_paths_hkg_bos if "AMS" in f[0]][0]

    all_paths_hkg_nyc = list(nx.all_simple_edge_paths(G, source="HKG", target="JFK", cutoff=2))
    path_hkg_ams_nyc = [f for f in all_paths_hkg_nyc if "AMS" in f[0]][0]

    # sea via Vietnam
    all_paths_hkhkg_usbos= list(nx.all_simple_edge_paths(G, source="HKHKG", target="USBOS", cutoff=2))
    path_hkhkg_usbos = [f for f in all_paths_hkhkg_usbos if "CNSHA" in f[0]][0]

    all_paths_hkhkg_usnyc= list(nx.all_simple_edge_paths(G, source="HKHKG", target="USNYC", cutoff=2))
    path_hkhkg_usnyc = [f for f in all_paths_hkhkg_usnyc if "SGSIN" in f[0]][0]

    #Combine paths
    subgraph = combine_shippings_routes_to_subgraph(G, [path_hkg_ams_lax, path_hkg_ams_nyc, path_hkhkg_usbos, path_hkhkg_usnyc])

    location_export_ports = ["HKG", "HKHKG"]
    location_import_ports = ["BOS", "JFK", "USBOS", "USNYC"]

    subgraph = remove_redudant_info_graph([subgraph])[0]

    # add attributes
    for n in subgraph.nodes():
        subgraph.nodes[n]["location"] = n
        if n in location_export_ports:
            subgraph.nodes[n]["node_type"] = "export_port"
            subgraph.nodes[n]["entity_type"] = "export_port_"+subgraph.nodes[n]["modality"]
            # for current visualization
            subgraph.nodes[n]["pos"] = (3, np.random.uniform(0, 10))

        elif n in location_import_ports:
            subgraph.nodes[n]["pos"] = (5, np.random.uniform(0, 10))
            subgraph.nodes[n]["node_type"] = "import_port"
            subgraph.nodes[n]["entity_type"] = "import_port_"+subgraph.nodes[n]["modality"]
            if subgraph.out_degree(n) > 0:
                subgraph.nodes[n]["node_functions"] = ["transit_port", "import_port"]
            else:
                subgraph.nodes[n]["node_type"] = "import_port"

        else:
            subgraph.nodes[n]["node_type"] = "transit_port"
            subgraph.nodes[n]["entity_type"] = "transit_port_"+subgraph.nodes[n]["modality"]
            subgraph.nodes[n]["pos"] = (4, np.random.uniform(0, 10))

    #Create dict mapping locations
    dict_node_location = {"HKG": "export_port_0",
                          "HKHKG": "export_port_1",
                          "BOS": "import_port_0",
                          "JFK": "import_port_1",
                          "USBOS": "import_port_2",
                          "USNYC": "import_port_3"}


    gt_graph = merge_graph_ports_with_generated_structure(graph_supply_chain, subgraph, dict_node_location)
    graphs = [gt_graph]

    df_gt_graph = calculate_network_topology_and_add_to_df(graphs)

    default_input_params = pd.read_excel(r"../input/default_input_params_actors_airsea.xlsx", sheet_name="actors")
    df_gt_graph = add_input_params_to_db_structures(df_gt_graph, default_input_params)

    with open(r"../data/Ground_Truth_Graph_Topology_DF_CNHK_USA.pkl", "wb") as file:
        pickle.dump(df_gt_graph, file)

    print("Graph Ground Truth is created and pickled")