"""
Created on: 3-5-2023 17:33

@author: IvS, Bruno Hermans
"""

import networkx as nx
import random
import time
import warnings
import pandas as pd
# import geopandas as gpd
# import rasterio
import numpy as np
# from shapely.geometry import Point, LineString
import logging
# from utilities.maps import (
#     get_urban_locations_tif,
#     get_port_locations,
#     generate_buffer_zone,
#     get_suitable_area,
#     get_distance_two_points,
#     get_shortest_sea_path,
# )
from structure_model_composer.helpers import find_overlaps



class StructureGenerator:
    """Baseclass that holds methods to generate random supply chain graphs.

    Args:
        res (dict): a loaded res.json file. File with all restricitions of the model composer
        seed (int, optional): Seed to set. Defaults to None.
        plot (bool, optional): To plot model structure or not. Defaults to False.

    Attributes:
        res (dict): Dictionary with all model composer restrictions
        seed (int): Seed to use when composing models
        plot (bool): Determines whether to plot the models graph or not.
        G (nx.Graph): Networkx Graph of model
        instance (dict): Dict with suppliers, manufacturers etc.
        nodes_per_type (dict): Dict with list of nodes per type.
        len_per_type (dict): Dict with number of suppliers, manufacturers etc.
        port_types (list): List with possible port types
        ports_supplier (gpd.GeoDataFrame): Dataset with possible supplier port locations.
        transit_ports (gpd.GeoDataFrame): Dataset with possible transit port locations.
        landuse_supplier (Rasterio object): Dataset with landuse of supplier
        border_supplier (gpd.GeoDataFrame): Dataset with administrative boundaries of supplier area.
        ports_receiver (gpd.GeoDataFrame): Dataset with ports on the receiving side of the supply chain. (import ports)
        landuse_receiver (Rasterio object): Dataset with landuse of the receiving side (import ports, wholesalers, retailers)
        border_receiver (gpd.GeoDataFrame): Dataset with administrative boundaries of area where retailers, wholesalers should be situated.
        sea_data_set (Rasterio object): Dataset with binary classification of sea and land.

    Methods:
        set_seed(self): Method that sets the seed of the model composer.
        generate_valid_structure(self): Method that generates a random graph, calls all other methods.
        create_instance(self): Generate a random number of entities per class type.
        validate_instance(self): Function that validates the number of nodes per type.
        create_ids(self): Function that creates ids for each node
        create_graph(self): Function that creates a graph
        create_graph_nodes(self): Function that creates networkx graph nodes.
        create_graph_edges(self): Function that creates networkx graph edges.
        validate_graph(self): Function that validates whether the graph is valid and complies to res.json.
        determine_edge_type(self): Determines if a link is a sea or land link.
        get_n_locations_needed(self): Set len_per_type attribute
        read_datasets(self): Reads all spatial datasets from res.json
        assign_coordinates(self): Method that computes coordinates for each port node.
        compute_locations_with_buffers(self): Method that computes coordinates for retailer, manufacturer, supplier and wholesaler nodes.
        set_coordinates(self): Method that assigns computed coordinates to node objects.
        compute_link_lengths(self): Method that computes the length of a link between two nodes.
        generate_spatial_dataset(self): Method that generates a shapefile of the generated model structure.
        create_graphset(self): Creates a set of non-isomorphic graphs.
        create_graphset_df(self): Creates a set of graphs with their centrality metrics and returns a dataframe.
    """

    def __init__(self, res, seed=None, plot=False):
        self.res = res
        self.seed = seed
        self.plot = plot
        self.G = None
        self.instance = None
        self.nodes_per_type = {}
        self.len_per_type = {}
        self.port_types = ["export_port", "transit_port", "import_port"]
        # datasets
        self.ports_supplier = None
        self.transit_ports = None
        self.landuse_supplier = None
        self.border_supplier = None
        self.ports_receiver = None
        self.landuse_receiver = None
        self.border_receiver = None
        self.sea_data_set = None
        self.set_seed()

    def set_seed(self):
        """Method that sets the seed of the generator

        Args:
            seed (int, optional): seed to set if none, class default is taken. Defaults to None.
        """
        logging.warning("seed set " + str(self.seed))
        if self.seed is None:
            self.random = random.Random()
        else:
            # np.random.seed(self.seed)
            # random.seed(self.seed)
            self.random = random.Random(self.seed)

    def generate_valid_structure(self, coordinates=True, return_spatial_dataset=False):
        """
        Function that generates a random graph that statisfies the contraints specified in a res.json
        """
        logging.info("Generating new structure..")
        start_time = time.time()
        # random.seed(self.seed)
        self.create_instance()
        self.validate_instance()
        self.create_graph()
        self.validate_graph()
        self.determine_edge_type()
        # if coordinates == True:
        #     self.read_datasets()
        #     self.set_coordinates()
        #     self.compute_link_lengths()
        # if return_spatial_dataset == True:
        #     self.generate_spatial_dataset()
        logging.warning(
            "Generating structure finished in %s" % (time.time() - start_time)
        )
        return self.G

    def create_instance(self):
        """
        Generate a random number of entities per class type.
        """
        res = self.res
        self.instance = {k: self.random.randint(1, v) for k, v in res["restrictions"].items()}
        return self.instance

    def validate_instance(self, instance=None, search=True):
        """Validate whether any structure can be generated or not.
        If the degree is unstatisfactory and the number of entities don't match,
        it is possible that the number of entities should be changed before establishing the
        edges between the nodes.

        Args:
            instance (dict): dictionary with number of nodes of each type
            search (bool, optional): If True, iterates untill true structure is found.
            If false, only returns wheter instance is true. Defaults to True.

        Returns:
            Bool or Dict: Returns true or false if search is true. Else returns valid instance
        """
        # Using either input instance or create one if it does not exist
        if instance == None and self.instance == None:
            self.create_instance()
            instance = self.instance
        elif instance != None:
            self.instance = instance
        elif self.instance != None:
            instance = self.instance

        # Reorganising restrictions and combining with instance
        valid = True
        res = self.res
        data = {}
        for key, value in instance.items():
            data[key.split("_", 1)[1][:-1]] = {}
        for key, value in instance.items():
            entity_type = key.split("_", 1)[1][:-1]
            min_inc_degree = res["chainage"][entity_type]["min_incoming_degree"]
            max_inc_degree = res["chainage"][entity_type]["max_incoming_degree"]
            min_out_degree = res["chainage"][entity_type]["min_outcoming_degree"]
            max_out_degree = res["chainage"][entity_type]["max_outcoming_degree"]
            data[entity_type]["n"] = value

            if min_inc_degree != -1:
                data[entity_type]["min_incoming_degree"] = value * min_inc_degree
            else:
                data[entity_type]["min_incoming_degree"] = -1

            if max_inc_degree != -1:
                data[entity_type]["max_incoming_degree"] = value * max_inc_degree
            else:
                data[entity_type]["max_incoming_degree"] = -1

            if min_out_degree != -1:
                data[entity_type]["min_outcoming_degree"] = value * min_out_degree
            else:
                data[entity_type]["min_outcoming_degree"] = -1

            if max_out_degree != -1:
                data[entity_type]["max_outcoming_degree"] = value * max_out_degree
            else:
                data[entity_type]["max_outcoming_degree"] = -1
        logging.debug(data)
        # Reorganisation complete, now checking for constraints.
        for key, value in data.items():
            ingoing_entity = res["chainage"][key]["ingoing_entity"]
            logging.debug("checking " + str(key))
            if ingoing_entity != "none":
                logging.debug("checking " + str(ingoing_entity) + " to " + str(key))
                my_max_in_degree = value["max_incoming_degree"]
                my_min_in_degree = value["min_incoming_degree"]
                others_max_out_degree = data[ingoing_entity]["max_outcoming_degree"]
                others_min_out_degree = data[ingoing_entity]["min_outcoming_degree"]
                others_out_degree_range = (others_min_out_degree, others_max_out_degree)
                my_in_degree_range = (my_min_in_degree, my_max_in_degree)
                logging.debug(str(others_out_degree_range) + str(my_in_degree_range))

                if others_out_degree_range == (-1, -1) and my_in_degree_range == (
                    -1,
                    -1,
                ):
                    logging.debug(str(ingoing_entity) + "has no restrictions")
                    pass
                else:
                    logging.debug("replacing -1 with infinity")
                    # indegree replacement
                    if my_in_degree_range == (-1, -1):
                        my_in_degree_range = (-float("inf"), float("inf"))
                    elif my_in_degree_range[0] == -1:
                        my_in_degree_range = (-float("inf"), my_in_degree_range[1])
                    elif my_in_degree_range[1] == -1:
                        my_in_degree_range = (my_in_degree_range[0], float("inf"))

                    # outdegree replacement
                    if others_out_degree_range == (-1, -1):
                        others_out_degree_range = (-float("inf"), float("inf"))
                    elif others_out_degree_range[0] == -1:
                        others_out_degree_range = (
                            -float("inf"),
                            others_out_degree_range[1],
                        )
                    elif others_out_degree_range[1] == -1:
                        others_out_degree_range = (
                            others_out_degree_range[0],
                            float("inf"),
                        )

                    logging.debug(
                        str(others_out_degree_range) + str(my_in_degree_range)
                    )
                    overlap = find_overlaps(
                        [my_in_degree_range[0], my_in_degree_range[1]],
                        [others_out_degree_range[0], others_out_degree_range[1]],
                    )
                    logging.debug("Overlap? " + str(overlap))
                    if overlap:
                        pass
                    else:
                        valid = False
                        if search == True:
                            instance = self.create_instance()
                            instance = self.validate_instance(instance)
        if search == True:
            self.instance = instance
            return instance
        else:
            return valid

    def create_ids(self):
        """
        Constructs an id for each individual entity.
        """
        nodes = {}
        if self.instance == None:
            instance = self.create_instance()
            instance = self.validate_instance(instance)
            logging.debug(instance)
        else:
            instance = self.instance
        id = 0
        for key, value in instance.items():
            entity_type = key.split("_", 1)[1]
            nodes[entity_type] = []
            for i in range(value):
                nodes[entity_type].append(id)
                id += 1
        self.ids = nodes
        return self.ids

    def create_graph(self):
        """
        Function that generates a random networkx graph that statisfies the given constraints.
        """
        self.create_ids()
        self.create_graph_nodes()
        self.create_graph_edges_2()

        return self.G

    def create_graph_nodes(self):
        """Creates nodes in nx.Graph."""
        res = self.res
        self.G = nx.DiGraph()
        # Add nodes
        for key, value in self.ids.items():
            for i in value:
                entity_type = key[:-1]
                max_o = res["chainage"][entity_type]["max_outcoming_degree"]
                min_o = res["chainage"][entity_type]["min_outcoming_degree"]
                max_i = res["chainage"][entity_type]["max_incoming_degree"]
                min_i = res["chainage"][entity_type]["min_incoming_degree"]
                inc_e = res["chainage"][entity_type]["ingoing_entity"]
                out_e = res["chainage"][entity_type]["outgoing_entity"]
                echelon = res["chainage"][entity_type]["echelon"]
                self.G.add_node(
                    i,
                    id=i,
                    entity_type=entity_type,
                    r_indegree=(min_i, max_i),
                    r_outdegree=(min_o, max_o),
                    indegree=0,
                    outdegree=0,
                    ingoing_entity=inc_e,
                    outgoing_entity=out_e,
                    echelon=echelon,
                    pos=(
                        res["chainage"][entity_type]["echelon"],
                        self.random.uniform(0, 10),
                    ),
                )

    def create_graph_edges_2(self):
        """Establishes graph edges. Can generate invalid and valid graphs.

        Returns:
            nx.graph: graph created/modified
        """
        # Add edges
        for i in range(len(self.G.nodes())):
            outgoing_entities = self.G.nodes()[i]["outgoing_entity"]
            entity_type = self.G.nodes()[i]["entity_type"]
            sim_entities = len([y for x, y in self.G.nodes(data=True) if y["entity_type"] == entity_type])

            if outgoing_entities != "none":
                if isinstance(outgoing_entities, str):
                    outgoing_entities = [outgoing_entities]
                edges_to_make = 0
                candidates = []
                for outgoing_entity in outgoing_entities:
                    forward_entities = [y for x, y in self.G.nodes(data=True) if y["entity_type"] == outgoing_entity]
                    suitable_entities = [z for z in forward_entities if
                                         (z["indegree"] <= z["r_indegree"][1] or z["r_indegree"][1] == -1)]
                    range_outdegree = self.G.nodes()[i]["r_outdegree"]

                    # Compute maximum and minimum number of entities of forward echelon
                    min_edges = sum([y["r_indegree"][0] for y in suitable_entities])
                    max_edges = sum([y["r_indegree"][1] for y in suitable_entities])
                    if min_edges > 0:
                        avg = max_edges / min_edges

                    if range_outdegree[1] == -1:
                        # If an unlimited number of outward edges can be made
                        if forward_entities[0]["r_indegree"][1] == -1:
                            # No restrictions on indegree forward entity
                            max_edges = len(suitable_entities)
                            if sim_entities == 1:
                                edges_to_make = len(suitable_entities)
                            else:
                                edges_to_make += self.random.randint(1, max_edges)
                        else:
                            if max_edges > len(suitable_entities):
                                max_edges = len(suitable_entities)
                                avg = max_edges / min_edges
                                edges_to_make += round(self.random.triangular(1, max_edges, avg))
                                logging.debug("Minimim, maximum", min_edges, max_edges)
                                logging.debug("edges to make, avg", edges_to_make, avg)
                            # Forward entity has limit
                            elif sim_entities == 1:
                                edges_to_make = len(suitable_entities)
                            else:
                                edges_to_make += round(self.random.triangular(1, max_edges, avg))
                    else:
                        if max_edges <= 0:
                            r_degree = self.G.nodes()[i]["r_outdegree"]
                            if sim_entities == 1:
                                edges_to_make = len(suitable_entities)
                            else:
                                max_edges_b = min(sim_entities * r_degree[1], max_edges)
                                a = round(min_edges / sim_entities)
                                edges_to_make += round(self.random.triangular(r_degree[0], r_degree[1], a))
                        else:
                            r_degree = self.G.nodes()[i]["r_outdegree"]
                            if sim_entities == 1:
                                edges_to_make = len(suitable_entities)
                            else:
                                max_edges_bf = min(sim_entities * r_degree[1], max_edges)
                                min_edges_bf = max(sim_entities * r_degree[0], min_edges)
                                a = round(min_edges_bf / sim_entities)
                                b = round(max_edges_bf / sim_entities)
                                if a != b:
                                    try:
                                        edges_to_make += self.random.randint(a, b)
                                    except:
                                        edges_to_make += a
                                elif a == b:
                                    edges_to_make += a

                self.G.nodes()[i]["edges_to_make"] = edges_to_make
                if (
                    len(suitable_entities) <= edges_to_make
                    and len(suitable_entities) >= self.G.nodes()[i]["r_outdegree"][0]
                ):
                    candidates = suitable_entities
                elif len(suitable_entities) > edges_to_make:
                    candidates = self.random.sample(suitable_entities, edges_to_make)
                else:
                    pass

                for k in candidates:
                    self.G.add_edge(i, k["id"])
                    self.G.nodes()[k["id"]]["indegree"] += 1
                    self.G.nodes()[i]["outdegree"] += 1

            else:
                self.G.nodes()[i]["edges_to_make"] = 0

        return self.G
    def create_graph_edges(self):
        """Establishes graph edges. Can generate invalid and valid graphs.

        Returns:
            nx.graph: graph created/modified
        """
        # Add edges
        for i in range(len(self.G.nodes())):
            outg = self.G.nodes()[i]["outgoing_entity"]
            entity_type = self.G.nodes()[i]["entity_type"]
            sim_entities = len(
                [
                    y
                    for x, y in self.G.nodes(data=True)
                    if y["entity_type"] == entity_type
                ]
            )
            if outg != "none":
                forward_entities = [
                    y for x, y in self.G.nodes(data=True) if y["entity_type"] == outg
                ]
                suitable_entities = [
                    z
                    for z in forward_entities
                    if (z["indegree"] <= z["r_indegree"][1] or z["r_indegree"][1] == -1)
                ]
                range_outdegree = self.G.nodes()[i]["r_outdegree"]

                # Compute maximum and minimum number of entities of forward echelon
                min_edges = sum([y["r_indegree"][0] for y in suitable_entities])
                max_edges = sum([y["r_indegree"][1] for y in suitable_entities])
                if min_edges > 0:
                    avg = max_edges / min_edges

                if range_outdegree[1] == -1:
                    # If an unlimited number of outward edges can be made
                    if forward_entities[0]["r_indegree"][1] == -1:
                        # No restrictions on indegree forward entity
                        max_edges = len(suitable_entities)
                        if sim_entities == 1:
                            edges_to_make = len(suitable_entities)
                        else:
                            edges_to_make = self.random.randint(1, max_edges)
                    else:
                        if max_edges > len(suitable_entities):
                            max_edges = len(suitable_entities)
                            avg = max_edges / min_edges
                            edges_to_make = round(
                                self.random.triangular(1, max_edges, avg)
                            )
                            logging.debug("Minimim, maximum", min_edges, max_edges)
                            logging.debug("edges to make, avg", edges_to_make, avg)
                        # Forward entity has limit
                        elif sim_entities == 1:
                            edges_to_make = len(suitable_entities)
                        else:
                            edges_to_make = round(
                                self.random.triangular(1, max_edges, avg)
                            )
                else:
                    if max_edges <= 0:
                        r_degree = self.G.nodes()[i]["r_outdegree"]
                        if sim_entities == 1:
                            edges_to_make = len(suitable_entities)
                        else:
                            max_edges_b = min(sim_entities * r_degree[1], max_edges)
                            a = round(min_edges / sim_entities)
                            edges_to_make = round(
                                self.random.triangular(r_degree[0], r_degree[1], a)
                            )
                    else:
                        r_degree = self.G.nodes()[i]["r_outdegree"]
                        if sim_entities == 1:
                            edges_to_make = len(suitable_entities)
                        else:
                            max_edges_bf = min(sim_entities * r_degree[1], max_edges)
                            min_edges_bf = max(sim_entities * r_degree[0], min_edges)
                            a = round(min_edges_bf / sim_entities)
                            b = round(max_edges_bf / sim_entities)
                            if a != b:
                                try:
                                    # print(a,b)
                                    edges_to_make = self.random.randint(a, b)
                                except:
                                    edges_to_make = a
                            elif a == b:
                                edges_to_make = a

                self.G.nodes()[i]["edges_to_make"] = edges_to_make
                if (
                    len(suitable_entities) <= edges_to_make
                    and len(suitable_entities) >= self.G.nodes()[i]["r_outdegree"][0]
                ):
                    candidates = suitable_entities
                elif len(suitable_entities) > edges_to_make:
                    candidates = self.random.sample(suitable_entities, edges_to_make)
                else:
                    pass

                for k in candidates:
                    self.G.add_edge(i, k["id"])
                    self.G.nodes()[k["id"]]["indegree"] += 1
                    self.G.nodes()[i]["outdegree"] += 1

            else:
                self.G.nodes()[i]["edges_to_make"] = 0

        return self.G

    def validate_graph(self, search=True):
        """Function that checks if the inputed graph statisfies the given constraints

        Args:
            search (bool, optional): to search for a new valid structure or not. Defaults to True.

        Returns:
            bool: if graph provided is valid or not.
        """
        G = self.G
        valid = True
        for i in range(len(G.nodes())):
            too_few_indegree = False
            too_much_indegree = False
            if G.nodes()[i]["r_indegree"] == (-1, -1):
                pass
            elif G.nodes()[i]["r_indegree"][0] == -1:
                if G.in_degree[i] > G.nodes()[i]["r_indegree"][1]:
                    logging.debug("Node " + str(i) + " has too much indegree")
                    too_much_indegree = True
                else:
                    pass
            elif G.nodes()[i]["r_indegree"][1] == -1:
                if G.in_degree[i] < G.nodes()[i]["r_indegree"][0]:
                    # print("Node " + str(i) + " has too few indegree")
                    too_few_indegree = True
                else:
                    pass
            elif G.in_degree[i] < G.nodes()[i]["r_indegree"][0]:
                # print("Node " + str(i) + " has too few indegree")
                too_few_indegree = True

            elif G.in_degree[i] > G.nodes()[i]["r_indegree"][1]:
                logging.debug("Node " + str(i) + " has too much indegree")
                too_much_indegree = True

            if too_few_indegree == True:
                try:
                    backward_entity_type = G.nodes()[i]["ingoing_entity"]
                    backward_entities = [
                        y
                        for x, y in G.nodes(data=True)
                        if y["entity_type"] == backward_entity_type
                    ]
                    suitable_entities = [
                        z
                        for z in backward_entities
                        if (
                            z["outdegree"] <= z["r_outdegree"][1]
                            or z["r_outdegree"][1] == -1
                        )
                    ]
                    G.add_edge(suitable_entities[0]["id"], i)
                except:
                    valid = False
                    logging.debug(
                        "Node " + str(i) + " has too much indegree, could not be fixed"
                    )

            if too_much_indegree == True:
                back_ward_entities = [x[0] for x in G.in_edges(i)]
                # to_remove = G.in_degree[i] - G.nodes()[i]['r_indegree'][1]
                for k in back_ward_entities:
                    if (
                        G.out_degree[k] > G.nodes()[k]["r_outdegree"][0]
                        and G.in_degree[i] <= G.nodes()[k]["r_indegree"][1]
                    ):
                        logging.debug("Edge ", k, "can be removed!")
                        G.remove_edge(k, i)
                    else:
                        valid = False

            # Check outdegree
            if G.nodes()[i]["r_outdegree"] == (-1, -1):
                pass
            elif G.nodes()[i]["r_outdegree"][0] == -1:
                if G.nodes()[i]["outdegree"] > G.nodes()[i]["r_outdegree"][1]:
                    logging.debug("Node " + str(i) + "has too much outdegree")
                    valid = False
                else:
                    pass
            elif G.nodes()[i]["r_outdegree"][1] == -1:
                if G.nodes()[i]["outdegree"] < G.nodes()[i]["r_outdegree"][0]:
                    logging.debug("Node " + str(i) + "has too few outdegree")
                    valid = False
                else:
                    pass
            elif G.nodes()[i]["outdegree"] < G.nodes()[i]["r_outdegree"][0]:
                logging.debug("Node " + str(i) + " has too few outdegree")
                valid = False
            elif G.nodes()[i]["outdegree"] > G.nodes()[i]["r_outdegree"][1]:
                logging.debug("Node " + str(i) + " has too much outdegree")
                valid = False

        if search == False:
            logging.debug("Structure is " + str(valid))
            return valid
        else:
            if valid == False:
                # If drawing edges failed, try agian.
                logging.debug("Retrying to create graph for this instance.")
                self.create_graph_nodes()
                self.create_graph_edges()
                self.validate_graph()
            else:
                return valid

    def determine_edge_type(self):
        """
        Function to assign link type to each edge of the Graph.
        Links between ports should be sealinks and links between other entities should be normal links
        """
        link_type = "link"
        nx.set_edge_attributes(self.G, link_type, "link_type")
        for edge in self.G.edges:
            from_entity = self.G.nodes()[edge[0]]["entity_type"]
            to_entity = self.G.nodes()[edge[1]]["entity_type"]

            if from_entity in self.port_types and to_entity in self.port_types:
                link_type = "sealink"
                self.G.edges[edge[0], edge[1]]["link_type"] = link_type

    def get_n_locations_needed(self, entity_types):
        """Function that set len_per_type atrribute.

        Args:
            entity_types (dict): list with all entity types

        Returns:
            int: summed number of entities of type(s) specified.
        """
        locations_needed = 0
        for i in entity_types:
            locations_needed += self.len_per_type[i]
        return locations_needed

    # def read_datasets(self):
    #     """Function that loads all spatial datasets to the object, to make sure that this happens only once."""
    #     datasets = self.res["datasets"]
    #     self.ports_supplier = gpd.read_file(datasets["ports_supplier"]).set_crs(
    #         "EPSG:4326", inplace=True
    #     )
    #     if "transit_ports" in datasets:
    #         self.transit_ports = gpd.read_file(datasets["transit_ports"]).set_crs(
    #             "EPSG:4326", inplace=True
    #         )
    #     else:
    #         self.transit_ports = gpd.read_file(datasets["ports_supplier"]).set_crs(
    #             "EPSG:4326", inplace=True
    #         )
    #     self.landuse_supplier = rasterio.open(datasets["landuse_supplier"])
    #     self.border_supplier = gpd.read_file(datasets["border_supplier"]).set_crs(
    #         "EPSG:4326", inplace=True
    #     )
    #     self.ports_receiver = gpd.read_file(datasets["ports_receiver"]).set_crs(
    #         "EPSG:4326", inplace=True
    #     )
    #     self.landuse_receiver = rasterio.open(datasets["landuse_receiver"])
    #     self.border_receiver = gpd.read_file(datasets["border_receiver"]).set_crs(
    #         "EPSG:4326", inplace=True
    #     )
    #     self.sea_data_set = rasterio.open(datasets["sea_data_set"])
    #
    # def assign_coordinates(self, points, locations_needed, entity_types, name=False):
    #     """Function that computes nodes of port locations.
    #
    #     Args:
    #         points (List): List of dictionaries. Each dictionary is a port location.
    #         locations_needed (int): number of locations needed.
    #         entity_types (_type_): List of entity types to be allocated
    #         name (bool, optional): unused parameter. Defaults to False.
    #     """
    #     coords_given = 0
    #     if isinstance(points[0], dict):
    #         for id, data in self.G.nodes(data=True):
    #             if (
    #                 data["entity_type"] in entity_types
    #                 and coords_given < locations_needed
    #             ):
    #                 self.G.nodes()[id]["coords"] = (
    #                     points[coords_given]["geometry"].x,
    #                     points[coords_given]["geometry"].y,
    #                 )
    #                 try:
    #                     self.G.nodes()[id]["location_name"] = points[coords_given][
    #                         "Name"
    #                     ]
    #                 except:
    #                     self.G.nodes()[id]["location_name"] = "Unknown"
    #                     warnings.warn("A selected port does not have a name")
    #                 try:
    #                     self.G.nodes()[id]["port_type"] = points[coords_given]["Type"]
    #                 except:
    #                     self.G.nodes()[id]["port_type"] = "Unknown"
    #                     warnings.warn("A selected port does not have a type")
    #                 coords_given += 1
    #     else:
    #         for id, data in self.G.nodes(data=True):
    #             if (
    #                 data["entity_type"] in entity_types
    #                 and coords_given < locations_needed
    #             ):
    #                 self.G.nodes()[id]["coords"] = (
    #                     points[coords_given].x,
    #                     points[coords_given].y,
    #                 )
    #                 coords_given += 1
    #
    # def compute_locations_with_buffers(
    #     self,
    #     next_entity_type,
    #     direction="forward",
    #     size=200000,
    #     increment=100000,
    #     border_path=None,
    #     landuse_dataset=None,
    # ):
    #     """Function that computes the location of retailers, manufacturers, wholesalers and suppliers.
    #
    #     Args:
    #         next_entity_type (str): Echelon to allocate. For example: 'manufacturers'
    #         direction (str, optional): Direction to operate. Defaults to "forward".
    #         size (int, optional): maximum area to draw buffer. Defaults to 200000.
    #         increment (int, optional): size to add if buffer is too small. Defaults to 100000.
    #         border_path (str, optional): path to borders dataset. Defaults to None.
    #         landuse_dataset (str, optional): path to landuse dataset. Defaults to None.
    #     """
    #     if border_path is None:
    #         border_path = self.border_receiver
    #     if isinstance(border_path, str):
    #         borders_dataset = gpd.read_file(border_path)
    #     else:
    #         borders_dataset = border_path
    #
    #     if landuse_dataset is None:
    #         landuse_dataset = self.landuse_receiver
    #
    #     for id, data in self.G.nodes(data=True):
    #         if data["entity_type"] == next_entity_type:
    #             # if next entity type is supplier and direction is forward,
    #             # location will be assigned for manufacturer
    #             if direction == "backward":
    #                 edges = self.G.in_edges(id)
    #                 entities = [j[0] for j in edges]
    #             if direction == "forward":
    #                 edges = self.G.out_edges(id)
    #                 entities = [j[1] for j in edges]
    #
    #             for entity in entities:
    #                 points_list = []
    #                 coords = self.G.nodes(data=True)[entity]["coords"]
    #                 point = Point(coords[0], coords[1])
    #                 points_list.append({"id": entity, "geometry": point})
    #             location_data = gpd.GeoDataFrame(points_list)
    #             location_data.set_crs("EPSG:4326", inplace=True)
    #             found_locations = False
    #             radius = size
    #             while found_locations == False:
    #                 radius += increment
    #                 buffers = generate_buffer_zone(location_data, size=radius)
    #                 buffers_geometry = buffers["geometry"].to_list()
    #                 intersections = []
    #                 for i in buffers_geometry:
    #                     for j in buffers_geometry:
    #                         intersections.append(i.intersects(j))
    #                 found_locations = all(intersections)
    #             intersection = buffers_geometry[0]
    #
    #             for i in buffers_geometry:
    #                 intersection = intersection.intersection(i)
    #             if intersection.intersects(borders_dataset["geometry"].iloc[0]):
    #                 intersection = intersection.intersection(
    #                     borders_dataset["geometry"].iloc[0]
    #                 )
    #             elif intersection.within(borders_dataset["geometry"].iloc[0]):
    #                 pass
    #             else:
    #                 # If port is outside administrative borders
    #                 while (
    #                     intersection.intersects(borders_dataset["geometry"].iloc[0])
    #                     == False
    #                 ):
    #                     radius += increment
    #                     buffers = generate_buffer_zone(location_data, size=radius)
    #                     buffers_geometry = buffers["geometry"].to_list()
    #                     intersections = []
    #                     for i in buffers_geometry:
    #                         for j in buffers_geometry:
    #                             intersections.append(i.intersects(j))
    #                     intersection = buffers_geometry[0]
    #                 logging.debug(
    #                     "Circles do not intersect with Geometry of the Netherlands!"
    #                 )
    #
    #             data = [{"id": id, "geometry": intersection}]
    #             intersection = gpd.GeoDataFrame(data)
    #             intersection.set_crs("EPSG:4326", inplace=True)
    #             if self.plot:
    #                 intersection.plot()
    #             # start_time = time.time()
    #             points_entity = get_urban_locations_tif(
    #                 clipping_region=intersection,
    #                 landuse_dataset=landuse_dataset,
    #                 borders=border_path,
    #                 n=1,
    #                 max_tries=100000,
    #                 plot=self.plot,
    #                 # seed=self.seed,
    #                 random_object=self.random,
    #             )
    #             # print("--- Location found in %s seconds ---" % (time.time() - start_time))
    #             # print("Location found, ", points_entity[0].x, points_entity[0].y)
    #             self.G.nodes()[id]["coords"] = (points_entity[0].x, points_entity[0].y)
    #
    # def set_coordinates(self):
    #     """Function that computes and sets the coordinates of all nodes"""
    #     # Create dictionary sorted by type of entity
    #     for id, data in self.G.nodes(data=True):
    #         if data["entity_type"] not in self.nodes_per_type:
    #             self.nodes_per_type[data["entity_type"]] = []
    #         self.nodes_per_type[data["entity_type"]].append(data)
    #
    #     # Create dict with len per type
    #     for key, value in self.nodes_per_type.items():
    #         self.len_per_type[key] = len(value)
    #
    #     # Generate coordinates for Vietnam Ports (export ports)
    #     vietnam_ports = ["export_port"]
    #     # locations_needed_vietnam = self.get_n_locations_needed(vietnam_ports)
    #     locations_needed_vietnam = self.instance["n_export_ports"]
    #     locations_vietnam = get_port_locations(
    #         self.ports_supplier, n=locations_needed_vietnam, seed=self.seed
    #     )
    #     self.assign_coordinates(
    #         locations_vietnam, locations_needed_vietnam, vietnam_ports
    #     )
    #
    #     # Generate coordinates for Transit ports
    #     transit_ports = ["transit_port"]
    #     # locations_needed_vietnam = self.get_n_locations_needed(vietnam_ports)
    #     locations_needed_transit = self.instance["n_transit_ports"]
    #     locations_transit = get_port_locations(
    #         self.transit_ports, n=locations_needed_transit, seed=self.seed
    #     )
    #     self.assign_coordinates(
    #         locations_transit, locations_needed_transit, transit_ports
    #     )
    #
    #     # Only select export ports for buffer zones
    #     ports_data = []
    #     for id, data in self.G.nodes(data=True):
    #         if data["entity_type"] == "export_port":
    #             ports_data.append(data)
    #     ports = pd.DataFrame(ports_data)
    #
    #     # Compute intersection between borders vietnam and buffers
    #     ports["geometry"] = ports["coords"].apply(lambda x: Point(x))
    #     ports["points"] = ports["geometry"]
    #     ports = gpd.GeoDataFrame(ports)
    #     print("Setting ports finished")
    #     # Assign manufacturer locations
    #     logging.info("assigning manufacturer locations")
    #     self.compute_locations_with_buffers(
    #         "manufacturer",
    #         direction="forward",
    #         size=self.res["buffers"]["buffer_manufacturer"],
    #         increment=self.res["buffers"]["increment"],
    #         border_path=self.border_supplier,
    #          landuse_dataset=self.landuse_supplier,
    #     )
    #
    #     # Assign supplier locations
    #     logging.info("assigning supplier locations")
    #     self.compute_locations_with_buffers(
    #         "supplier",
    #         direction="forward",
    #         size=self.res["buffers"]["buffer_supplier"],
    #         increment=self.res["buffers"]["increment"],
    #         border_path=self.border_supplier,
    #         landuse_dataset=self.landuse_supplier,
    #     )
    #     logging.info("assigning import port locations")
    #     # Assign import_ports locations
    #     benelux_ports = ["import_port"]
    #     # locations_needed_benelux = self.get_n_locations_needed(benelux_ports)
    #     locations_needed_benelux = self.instance["n_import_ports"]
    #     logging.debug(locations_needed_benelux)
    #     locations_benelux = get_port_locations(
    #         self.ports_receiver, n=locations_needed_benelux, seed=self.seed
    #     )
    #     self.assign_coordinates(
    #         locations_benelux, locations_needed_benelux, benelux_ports
    #     )
    #
    #     # Assign location for wholesaler
    #     logging.info("Assigning wholesaler locations")
    #     self.compute_locations_with_buffers(
    #         "wholesaler",
    #         direction="backward",
    #         size=self.res["buffers"]["buffer_wholesaler"],
    #         increment=self.res["buffers"]["increment"],
    #         border_path=self.border_receiver,
    #         landuse_dataset=self.landuse_receiver,
    #     )
    #
    #     # Assign retailer locations
    #     logging.info("assigning retailer locations")
    #     self.compute_locations_with_buffers(
    #         "retailer",
    #         direction="backward",
    #         size=self.res["buffers"]["buffer_retailer"],
    #         increment=self.res["buffers"]["increment"],
    #         border_path=self.border_receiver,
    #         landuse_dataset=self.landuse_receiver,
    #     )
    #
    # def compute_link_lengths(self):
    #     """Function that assigns lengths to edges and geographical shapes of the edges"""
    #     for edge in self.G.edges(data=True):
    #         point1 = self.G.nodes()[edge[0]]["coords"]
    #         point2 = self.G.nodes()[edge[1]]["coords"]
    #         link_type = edge[2]["link_type"]
    #         logging.debug("points for line", point1, point2)
    #         if link_type == "sealink":
    #             shape, length = get_shortest_sea_path(point1, point2, self.sea_data_set)
    #         if link_type == "link":
    #             shape, length = get_distance_two_points(point1, point2)
    #         self.G.edges()[edge[0], edge[1]]["shape"] = shape
    #         self.G.edges()[edge[0], edge[1]]["weight"] = length
    #
    # def generate_spatial_dataset(self):
    #     """Method that can be used to generate a spatial dataset of the generated structure."""
    #     dataset_supplychain = []
    #     # Add node data to spatial dataset
    #     for id, data in self.G.nodes(data=True):
    #         spatial_data = {"id": id, "entity_type": data["entity_type"]} | {
    #             "geometry": Point(data["coords"][0], data["coords"][1])
    #         }
    #         dataset_supplychain.append(spatial_data)
    #     # Add edge data to spatial dataset
    #     for left_edge, right_edge, data in self.G.edges(data=True):
    #         shape = data["shape"]
    #         spatial_data = (
    #             {"id": str((left_edge, right_edge))}
    #             | {"entity_type": data["link_type"]}
    #             | {"geometry": shape}
    #         )
    #         dataset_supplychain.append(spatial_data)
    #     supply_chain = gpd.GeoDataFrame(dataset_supplychain)
    #     supply_chain.to_file("entities.geojson")

    def create_graphset(self, coordinates=False, n: int = 1000):
        """Creates a set of non isomorphic graphs.

        Args:
            coordinates (bool): whether to assign the coordinates or not. Defaults to False.
            n (int, optional): Number of graphs to create (initially). Defaults to 1000.
        """

        graphs = []
        for i in range(n):
            G = self.generate_valid_structure(coordinates=coordinates)
            graphs.append(G)

        for i in graphs:
            for j in graphs:
                if i == j:
                    pass
                else:
                    isomorphic = nx.is_isomorphic(i, j)
                    if isomorphic:
                        graphs.remove(j)
        return graphs

    def create_graphset_df(self, coordinates=False, n: int = 1000):
        """Function that creates a set of graphs with centrality metrics.

        Args:
            coordinates (bool, optional): Whether to generate a structure with or without coordinates. Defaults to False.
            n (int, optional): Number of graphs to create. Defaults to 1000.

        Returns:
            pd.DataFrame: DataFrame of graphs including betweenness, n_nodes, betweenness and centrality metrics.
        """
        graphs = self.create_graphset(coordinates=coordinates, n=n)
        data = []
        index = 0
        for i in graphs:
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
                                i, weight="weight"
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
