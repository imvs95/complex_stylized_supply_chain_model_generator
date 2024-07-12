"""
Created on: 25-4-2023 14:23

@author: IvS
"""
import pickle
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd

import networkx as nx

from structure_model_composer.sample import save_skeletons_batch, read_skeleton_data_base, create_graph_hashes
from structure_model_composer.process_results import draw_graph

np.random.seed(1)
def select_random_shipping_path(graph, source, target, cutoff=None):
    """Select shipping route for one source and target using the cutoff. Cutoff is
    the depth of the search - so cutoff of 2 means that there are three ports included."""
    all_paths = list(nx.all_simple_edge_paths(graph, source=source, target=target, cutoff=cutoff))
    if len(all_paths) == 1:
        random_path_edges = all_paths[0]
    else:
        rand_indx = np.random.randint(0, len(all_paths))
        random_path_edges = all_paths[rand_indx]
    return random_path_edges

def combine_shippings_routes_to_subgraph(graph, list_shipping_paths):
    """Combine all shipping routes to one subgraph"""

    subgraphs = []
    for path in list_shipping_paths:
        subgraph_path = graph.edge_subgraph(path)
        subgraphs.append(subgraph_path)

    combined_graph = nx.compose_all(subgraphs)
    return combined_graph

def randomly_select_import_ports(destination_ports, import_ports):
    """ Randomly select import ports. Ensure that at least one airport and one seaport is selected. This function makes
    sure that the import ports can be reached from the same type of the export ports. """

    import_air = [item for item in destination_ports if len(item) == 3]
    import_sea = [item for item in destination_ports if len(item) != 3]

    # select at least one from both
    location_air = np.random.choice(import_air, 1, replace=False)
    location_sea = np.random.choice(import_sea, 1, replace=False)

    location_import_ports = np.append(location_air, location_sea)

    total_ports_left = len(import_ports) - 2
    if total_ports_left > 0:
        other_location_import_ports = np.random.choice(destination_ports, total_ports_left, replace=False)
        location_import_ports = np.append(location_import_ports, other_location_import_ports)
    else:
        pass
    return location_import_ports

    # # if there are more import ports
    # location_import_ports = np.random.choice(destination_ports, len(import_ports), replace=False)
    # if any(len(item) != 3 for item in location_import_ports) or any(len(item) != 4 for item in location_import_ports):
    #     return randomly_select_import_ports(destination_ports, import_ports)
    # else:
    #     return location_import_ports

def determine_port_locations_and_create_subgraph(structure, graph_opensource, origin_ports, destination_ports):
    """ Randomly determine the location of the ports for the graph structure and create a subgraph with the ports,
    and the shipping routes. This function makes sure that seaports can only access seaports, and similar for airports.

    Parameters:
        structure (dict): structure of the graph (based on the random generator - db)
        graph_opensource (networkx.Graph): large graph with the shipping routes, based on open source data
        origin_ports (list): list of origin ports
        destination_ports (list): list of destination ports

    Returns:
        new_subgraph_ports (networkx.Graph): subgraph with the ports and shipping routes
        dict_nodes_locations (dict): dictionary with the locations of the ports
    """
    graph_structure = structure["graph"]
    # Get only the graph of the ports
    subgraph_ports = nx.DiGraph(graph_structure.subgraph([n for n, d in graph_structure.nodes(data=True)
                                               if "port" in d["entity_type"]]))

    export_ports = [n for n, d in subgraph_ports.nodes(data=True) if "export_port" in d["entity_type"]]
    import_ports = [n for n, d in subgraph_ports.nodes(data=True) if "import_port" in d["entity_type"]]

    # randomly select export port (sea and airports)
    location_export_ports = np.random.choice(origin_ports, len(export_ports), replace=False)
    if any(len(item) == 3 for item in location_export_ports):
        if len(import_ports) == 1:
            # just randomly choose if it's going to be an airport or seaport
            location_import_ports = np.random.choice(destination_ports, 1, replace=False)
        else:
            # ensure at least one destination airport and at least one destination seaport
            location_import_ports = randomly_select_import_ports(destination_ports, import_ports)
    elif all(len(item) == 3 for item in location_export_ports):
        # all are airports
        destination_ports = [item for item in destination_ports if len(item) == 3]
        try:
            location_import_ports = np.random.choice(destination_ports, len(import_ports), replace=False)
        except ValueError:
            # more ports than possible
            nodes_to_remove = import_ports[len(destination_ports):]
            import_ports = import_ports[:len(destination_ports)]
            subgraph_ports.remove_nodes_from(nodes_to_remove)
            location_import_ports = np.random.choice(destination_ports, len(import_ports), replace=False)
    else:
        # none is airport
        # remove all elements with three letters to only ensure sea ports
        destination_ports = [item for item in destination_ports if len(item) != 3]
        try:
            location_import_ports = np.random.choice(destination_ports, len(import_ports), replace=False)
        except ValueError:
            # more ports than possible
            nodes_to_remove = import_ports[len(destination_ports):]
            import_ports = import_ports[:len(destination_ports)]
            subgraph_ports.remove_nodes_from(nodes_to_remove)
            location_import_ports = np.random.choice(destination_ports, len(import_ports), replace=False)

    dict_nodes_locations = defaultdict()
    for i, l in zip(export_ports, list(location_export_ports)):
        subgraph_ports.nodes[i]["geo_location"] = l
        dict_nodes_locations[l] = i

    for i, l in zip(import_ports, list(location_import_ports)):
        subgraph_ports.nodes[i]["geo_location"] = l
        dict_nodes_locations[l] = i

    all_shipping_paths = []
    dict_exp_dest = defaultdict(list)
    for exp_idx in export_ports:
        # we use the transit ports to determine the "end" nodes, not the number of transit nodes - this is random
        descendants = nx.descendants(subgraph_ports, exp_idx)
        final_successors = [node for node in descendants if subgraph_ports.out_degree(node) == 0]
        for dest_idx in final_successors:
            dict_exp_dest[subgraph_ports.nodes[exp_idx]["geo_location"]].append(subgraph_ports.nodes[dest_idx]["geo_location"])
            if len(subgraph_ports.nodes[exp_idx]["geo_location"]) == 3 and len(subgraph_ports.nodes[dest_idx]["geo_location"]) == 3:
                # for airports -- else too long
                random_cutoff = np.random.randint(1, 4)
            else:
                random_cutoff = np.random.randint(1, 5)
            try:
                if len(subgraph_ports.nodes[exp_idx]["geo_location"]) != len(
                        subgraph_ports.nodes[dest_idx]["geo_location"]):
                    continue
                else:
                    shipping_path = select_random_shipping_path(graph_opensource, subgraph_ports.nodes[exp_idx]["geo_location"],
                                                                subgraph_ports.nodes[dest_idx]["geo_location"], cutoff=random_cutoff)
                    all_shipping_paths.append(shipping_path)
            except ValueError:
                try:
                    if len(subgraph_ports.nodes[exp_idx]["geo_location"]) == 3 and len(
                            subgraph_ports.nodes[dest_idx]["geo_location"]) == 3:
                        #Cutoff is too small for airports (5 is too big)
                        shipping_path = select_random_shipping_path(graph_opensource,
                                                                    subgraph_ports.nodes[exp_idx]["geo_location"],
                                                                    subgraph_ports.nodes[dest_idx]["geo_location"],
                                                                    cutoff=4)

                    else:
                        #Cutoff is too small
                        shipping_path = select_random_shipping_path(graph_opensource,
                                                                    subgraph_ports.nodes[exp_idx]["geo_location"],
                                                                    subgraph_ports.nodes[dest_idx]["geo_location"],
                                                                    cutoff=5)
                    all_shipping_paths.append(shipping_path)
                except ValueError:
                    # route does not exist
                    continue
    if len(all_shipping_paths) > 0:
        new_subgraph_ports = combine_shippings_routes_to_subgraph(graph_opensource, all_shipping_paths)
    else:
        # no shipping paths - because no feasible paths between origin and destination
        all_shipping_paths = []
        dict_exp_dest = defaultdict(list)
        for exp_idx in export_ports:
            # we use the transit ports to determine the "end" nodes, not the number of transit nodes - this is random
            descendants = nx.descendants(subgraph_ports, exp_idx)
            final_successors = [node for node in descendants if subgraph_ports.out_degree(node) == 0]
            for dest_idx in final_successors:
                all_paths_from_source = nx.single_source_shortest_path(graph_opensource,
                                                                    source=subgraph_ports.nodes[exp_idx]["geo_location"])

                all_paths_from_source = [tuple(v) for k, v in all_paths_from_source.items() if len(v) > 1]
                if len(all_paths_from_source) == 1:
                    shipping_path = all_paths_from_source[0]
                else:
                    rand_indx = np.random.randint(0, len(all_paths_from_source))
                    shipping_path = all_paths_from_source[rand_indx]

                # update all items
                new_dest = shipping_path[-1]
                dict_exp_dest[subgraph_ports.nodes[exp_idx]["geo_location"]].append(
                    new_dest)

                original_location = subgraph_ports.nodes[dest_idx]["geo_location"]
                location_import_ports = [new_dest if item == original_location
                                         else item for item in location_import_ports]
                subgraph_ports.nodes[dest_idx]["geo_location"] = shipping_path[-1]
                dict_nodes_locations[new_dest] = dest_idx

                # reformat with multiple stops
                edges_shipping_path = [(shipping_path[i], shipping_path[i + 1]) for i in range(len(shipping_path) - 1)]
                for edge in edges_shipping_path:
                    all_shipping_paths.append([edge])

        # to ensure the other ports are removed
        get_locations_in_graph = [attr["geo_location"] for n, attr in
                                  subgraph_ports.nodes(data=True) if "geo_location" in attr]
        dict_nodes_locations = {k: v for k, v in dict_nodes_locations.items() if k in get_locations_in_graph}

        # make subgraph
        new_subgraph_ports = combine_shippings_routes_to_subgraph(graph_opensource, all_shipping_paths)

    # add attributes
    for n in new_subgraph_ports.nodes():
        new_subgraph_ports.nodes[n]["location"] = n
        if n in location_export_ports:
            new_subgraph_ports.nodes[n]["node_type"] = "export_port"
            new_subgraph_ports.nodes[n]["entity_type"] = "export_port_"+new_subgraph_ports.nodes[n]["modality"]
            new_subgraph_ports.nodes[n]["possible_final_destination"] = dict_exp_dest[n]
            # for current visualization
            new_subgraph_ports.nodes[n]["pos"] = (3, np.random.uniform(0, 10))

        elif n in location_import_ports:
            new_subgraph_ports.nodes[n]["pos"] = (5, np.random.uniform(0, 10))
            new_subgraph_ports.nodes[n]["node_type"] = "import_port"
            new_subgraph_ports.nodes[n]["entity_type"] = "import_port_"+new_subgraph_ports.nodes[n]["modality"]
            if new_subgraph_ports.out_degree(n) > 0:
                new_subgraph_ports.nodes[n]["node_functions"] = ["transit_port", "import_port"]
            else:
                new_subgraph_ports.nodes[n]["node_type"] = "import_port"

        else:
            new_subgraph_ports.nodes[n]["node_type"] = "transit_port"
            new_subgraph_ports.nodes[n]["entity_type"] = "transit_port_"+new_subgraph_ports.nodes[n]["modality"]
            new_subgraph_ports.nodes[n]["pos"] = (4, np.random.uniform(0, 10))

    return new_subgraph_ports, dict_nodes_locations

def add_plot_graph_positions(graph, ports):
    """ Add the positions of the ports to the graph for plotting the graph with the positions."""
    # Plot nodes based on location
    for node, attributes in graph.nodes(data=True):
        try:
            if attributes["modality"] == "sea":
                attributes["latitude"] = ports[node]["LocationLatitude"]
                attributes["longitude"] = ports[node]["LocationLongitude"]
            else:
                continue
        except KeyError:
            raise Warning("Port {0} cannot be found".format(node))

    # position_ports = {k: (attr["longitude"], attr["latitude"]) for k, attr in graph.nodes(data=True)}
    #
    # plt.figure(figsize=(20, 10))
    # nx.draw_networkx(graph, pos=position_ports)
    # plt.show()

def merge_graph_ports_with_generated_structure(generated_structure, graph_ports, mapping_locations):
    """ Merge the generated structure (from the database) with the graph of the ports. This is done by combining the two graphs and
    relabeling the nodes to the correct location.

    Parameters:
        generated_structure (networkx.Graph): generated structure
        graph_ports (networkx.Graph): graph with the ports
        mapping_locations (dict): dictionary with the mapping of the locations to the nodes

    Returns:
        combined_graph_name (networkx.Graph): combined graph with the structure and the ports"""
    graph_no_ports = generated_structure.subgraph([n for n, d in generated_structure.nodes(data=True)
                              if "port" not in d["entity_type"]])
    all_graphs = nx.compose_all([graph_no_ports, graph_ports])
    combined_graph = nx.relabel_nodes(all_graphs, mapping_locations)

    #Find edges that are in the generated structure but not in the combined graph
    edges_to_add = [(l, r, d) for l, r, d in generated_structure.edges(data=True) if
             (l, r) not in combined_graph.edges() and l in combined_graph.nodes()
             and r in combined_graph.nodes()]

    combined_graph.add_edges_from(edges_to_add)

    combined_graph_name = nx.relabel_nodes(combined_graph, {v: k for k, v in mapping_locations.items()})

    #Remove nodes without any outgoing links, except end customer
    nodes_to_remove = [node for node, out_degree in combined_graph_name.out_degree() if out_degree == 0 and
                       combined_graph_name.nodes[node].get('entity_type') != 'end_customer']
    combined_graph_name.remove_nodes_from(nodes_to_remove)

    return combined_graph_name

def add_random_distance_to_supply_demand_side(graph_supply_chain, constraints):
    """ Add random distances to the edges of the graph. The distance is based on the constraints of the connection
    between the nodes. The constraints are based on the entity type of the nodes.

    Parameters:
        graph_supply_chain (networkx.Graph): graph with the supply chain
        constraints (dict): dictionary with the constraints per connection
    Returns:
        graph_supply_chain (networkx.Graph): graph with the distances added"""

    edges_to_set_distance = [(l, r, d) for l, r, d in graph_supply_chain.edges(data=True)
                             if len(d.keys()) == 1]

    constraints = {eval(k): v for k,v in constraints.items()}

    for edge in edges_to_set_distance:
        (o, d, attr) = edge
        node_connection = (graph_supply_chain.nodes[o]["entity_type"], graph_supply_chain.nodes[d]["entity_type"])
        if "end_customer" in node_connection:
            random_distance = constraints[node_connection]
        else:
            random_distance = np.random.randint(*constraints[node_connection])
        graph_supply_chain.edges[(o, d)]["distance_km"] = random_distance

    return graph_supply_chain

def calculate_network_topology_and_add_to_df(list_graphs):
    """ Calculate the network topology of the graphs and add this to a dataframe. The dataframe contains the index of the
    graph, the number of edges, the number of nodes, the betweenness centrality, the degree centrality, and the closeness centrality.
    """
    data = []
    index = 0
    for i in list_graphs:
        index += 1
        data.append(
            {
                "index": index,
                "graph": i,
                "edges": len(i.edges),
                "nodes": len(i.nodes),
                "betweenness": np.mean(
                    [
                        value
                        for key, value in nx.betweenness_centrality(
                        i,
                    ).items()
                    ]
                ),
                "degree_centrality": np.mean(
                    [value for key, value in nx.degree_centrality(i).items()]
                ),
                "closeness_centrality": np.mean(
                    [value for key, value in nx.closeness_centrality(i).items()]
                ),
            }
        )

    return pd.DataFrame(data)

def plot_graph_locations_with_pos(graph):
    """Plot the graph with the positions of the nodes"""
    #plot graph
    node_positions = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, with_labels=True, pos=node_positions)
    plt.show()

def remove_redudant_info_graph(list_graphs):
    """This function removes the redundant information from a graph. We only keep the values that are necessary for the
    simulation model. The keys that are kept are defined in the function."""

    keys_edges_to_keep = set(["frequency", "median_time_minutes", "distribution", "modality", "distance_km",
                              "other_modality"])
    keys_nodes_to_keep = set(["latitude", "longitude", "processing_distribution_full", "processing_distribution",
                              "location", "node_type", "entity_type", "pos", "modality",
                              "other_modality"])

    for g in list_graphs:
        for (o, d, attr) in g.edges(data=True):
            if isinstance(o, (str, np.str_)) and isinstance(d, (str, np.str_)):
                attr_to_remove_edge = [key for key in attr if key not in keys_edges_to_keep]
                for key in attr_to_remove_edge:
                    g.edges[(o, d)].pop(key)
            else:
                continue

        for (n, attr) in g.nodes(data=True):
            if isinstance(n, (str, np.str_)):
                attributes_to_remove = [key for key in attr if key not in keys_nodes_to_keep]
                for key in attributes_to_remove:
                    g.nodes[n].pop(key)
            else:
                continue
    return list_graphs


if __name__ == "__main__":
    # Open input data
    with open(r"../input/graph_networkx_CNHK_USA_FINAL_route_with_sea.pkl", 'rb') as f:
        G = pickle.load(f)

    with open(r"../input/list_origin_cnhk_us_final.pkl", "rb") as origin:
        list_origin = pickle.load(origin)
        # add air origin based on list CH/HK
        list_origin.extend(["HKG", "NGB", "MFM", "SZX", "SHA"])

    with open(r"../input/list_destination_cnhk_us_final.pkl", "rb") as dest:
        list_dest = pickle.load(dest)
        # add air
        list_dest.extend(["JFK", "BOS", "IAH", "LAX", "LGB", "SEA"])

    # Import ports data
    f = open("../input/msc_route_country_port_codes.json")
    port_locs = json.load(f)

    # with open("../input/distance_constraints_per_connections.pkl", "rb") as f:
    #     distance_constraints_per_connections = pickle.load(f)
    with open("../input/distance_constraints_per_connections_cnhk_us.json", "r") as f:
        distance_constraints_per_connections = json.load(f)

    # Open database of randomly generated structures
    db_generated_structures = read_skeleton_data_base(r"./databases/random_hpc_10_cnhk_usa")

    # Create new graph with locations
    graphs = []
    for _, row in db_generated_structures.iterrows():
        graph_ports, dict_nodes_location = determine_port_locations_and_create_subgraph(row, G, list_origin, list_dest)

        add_plot_graph_positions(graph_ports, port_locs)

        full_graph_with_ports = merge_graph_ports_with_generated_structure(row["graph"], graph_ports, dict_nodes_location)

        #For now, we choose to do this random. We can make this more realistic using shapefiles (Bruno), and exact
        #coordinates. And even use Google Maps/OSM/Hanna thesis for actual locations.

        final_graph = add_random_distance_to_supply_demand_side(full_graph_with_ports, distance_constraints_per_connections)

        # Add correct attributes
        for n, d in final_graph.nodes(data=True):
            if "node_type" not in d:
                d["node_type"] = d["entity_type"]
            else:
                continue

        graphs.append(final_graph)

    print("All graphs have been transformed!")
    df_graphs = calculate_network_topology_and_add_to_df(graphs)
    df_graphs = create_graph_hashes(df_graphs)
    # Save dataframe to database
    save_skeletons_batch(df_graphs, dbname="test_location_graphs_10")
    print("Graphs are saved to db")


