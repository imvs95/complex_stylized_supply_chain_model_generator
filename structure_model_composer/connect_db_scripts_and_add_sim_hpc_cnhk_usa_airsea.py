"""
Created on: 26-6-2023 11:31

@author: IvS
"""
import logging
import json
import pickle
import pandas as pd
import datetime
import sys
import numpy as np

from structure_model_composer.create_structure_with_db import fill_db
from structure_model_composer.sample import save_skeletons_batch, read_skeleton_data_base, create_graph_hashes

from structure_model_composer.set_locations_of_structures_airsea import determine_port_locations_and_create_subgraph, \
    add_plot_graph_positions, merge_graph_ports_with_generated_structure, add_random_distance_to_supply_demand_side, \
    calculate_network_topology_and_add_to_df, remove_redudant_info_graph

from structure_model_composer.preprocessing_graph import remove_edges_to_self

from utils.configure_input_data_graph_airsea import add_input_params_to_db_structures

if __name__ == "__main__":
    db_random = "random_hpc_50000_cnhk_usa_airsea"
    db_location = "location_model_hpc_50000_cnhk_usa_airsea"

    constraints_file = "../input/constraints_sc_nodoubleinput_cnhk_usa.json"

    # ## Create random structures
    logging.basicConfig(level=logging.INFO)
    fill_db(n=200, n_structures=250, db_name=r"./databases/" + db_random, constraints_file=constraints_file)
    # 200, 250

    ## Set structure
    with open(r"../input/graph_networkx_CNHK_USA_FINAL_route_with_sea.pkl", 'rb') as f:
        G = pickle.load(f)

    # Preprocessing Graph
    G = remove_edges_to_self(G)

    with open(r"../input/list_origin_cnhk_us_final.pkl", "rb") as origin:
        list_origin = pickle.load(origin)
        list_origin.extend(["HKG", "NGB", "MFM", "SZX", "SHA"])

    with open(r"../input/list_destination_cnhk_us_final.pkl", "rb") as dest:
        list_dest = pickle.load(dest)
        list_dest.extend(["JFK", "BOS", "IAH", "LAX", "LGB", "SEA"])

    # Import ports data
    f = open("../input/msc_route_country_port_codes.json")
    port_locs = json.load(f)

    with open("../input/distance_constraints_per_connections_cnhk_us.json", "r") as f:
        distance_constraints_per_connections = json.load(f)

    # Open database of randomly generated structures
    db_generated_structures = read_skeleton_data_base(r"./databases/" + db_random)

    print(f"{datetime.datetime.now():%Y-%m-%d}" + " Transformation has begun")

    # Create new graph with locations
    graphs = []
    for _, row in db_generated_structures.iterrows():
        graph_ports, dict_nodes_location = determine_port_locations_and_create_subgraph(row, G, list_origin,
                                                                                        list_dest)

        add_plot_graph_positions(graph_ports, port_locs)

        full_graph_with_ports = merge_graph_ports_with_generated_structure(row["graph"], graph_ports,
                                                                           dict_nodes_location)

        # For now, we choose to do this random. We can make this more realistic using shapefiles (Bruno), and exact
        # coordinates. And even use Google Maps/OSM/Hanna thesis for actual locations.

        final_graph = add_random_distance_to_supply_demand_side(full_graph_with_ports,
                                                                distance_constraints_per_connections)

        # Add correct attributes
        for n, d in final_graph.nodes(data=True):
            if "node_type" not in d:
                d["node_type"] = d["entity_type"]
            else:
                continue

        graphs.append(final_graph)

        if row.name % 1000 == 0:
            print(f"{datetime.datetime.now():%Y-%m-%d}" + " {0} graphs have been transformed".format(row.name + 1))
            continue

    del G
    del db_generated_structures

    print(f"{datetime.datetime.now():%Y-%m-%d}" + " All graphs have been transformed!")
    graphs = remove_redudant_info_graph(graphs)
    # calculate size of graphs
    mem = 0
    for i in graphs:
        graph_mem = np.sum([sys.getsizeof(e) for e in i.edges]) + np.sum([sys.getsizeof(n) for n in i.nodes])
        mem += graph_mem

    print(f"{datetime.datetime.now():%Y-%m-%d}" + " Size of graph is {0} MB".format(mem / 1024 / 1024))

    df_graphs = calculate_network_topology_and_add_to_df(graphs)
    df_graphs = create_graph_hashes(df_graphs)
    print(f"{datetime.datetime.now():%Y-%m-%d}" + " Network topologies are calculated")

    # Save dataframe to database
    # save_skeletons_batch(df_graphs, dbname=r"./databases/"+db_location)
    # logging.info("Graphs are saved to db")
    #
    # # Add parameters
    # db_structures = read_skeleton_data_base(r"./structure_model_composer/databases"+db_location)

    db_structures = df_graphs
    default_input_params = pd.read_excel(r"../input/default_input_params_actors_airsea.xlsx", sheet_name="actors")

    new_db = add_input_params_to_db_structures(db_structures, default_input_params)
    with open(r"../data/Ground_Truth_Graph_Topology_DF_CNHK_USA.pkl", "rb") as gt:
        ground_truth = pickle.load(gt)
        ground_truth["hash"] = "ground_truth"
    new_db = pd.concat([new_db, ground_truth]).reset_index(drop=True)

    # Sort for optimization
    network_topology = "betweenness"
    dict_db_sort_topology = new_db.sort_values(network_topology).reset_index(drop=True).to_dict("index")

    with open(r"../data/{0}_{1}.pkl".format(db_location, network_topology), "wb") as file:
        pickle.dump(dict_db_sort_topology, file)

    print(f"{datetime.datetime.now():%Y-%m-%d}" + " Graphs are sorted in dictionary on {0} and saved to pickle".format(
        network_topology))
