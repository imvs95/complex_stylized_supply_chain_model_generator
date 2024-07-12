"""
Created on: 16-1-2023 15:48

@author: IvS
"""

import itertools
import random
import statistics
import numpy as np
import pandas as pd
import time
import copy

from pydsol.core.model import DSOLModel

from sim_model_elements.entities import RawMaterials, Batch, Parcel, Product, Container
from sim_model_elements.supplier import Supplier
from sim_model_elements.manufacturer import Manufacturer
from sim_model_elements.wholesales_consolidator import WholesalesConsolidator
from sim_model_elements.port import Port, ImportPort
from sim_model_elements.airport import Airport, ImportAirport
from sim_model_elements.wholesales_distributor import WholesalesDistributor
from sim_model_elements.retailer import Retailer
from sim_model_elements.endcustomer import EndCustomer
from pydsol.model.link import Link
from sim_model_elements.link_airsea import SeaTimeLink
from sim_model_elements.link_intermodal import IntermodalTimeLink, DistanceLink

from sim_model_elements.vehicles.large_truck import LargeTruck
from sim_model_elements.vehicles.small_truck import SmallTruck
from sim_model_elements.vehicles.feeder import Feeder
from sim_model_elements.vehicles.vessel import Vessel
from sim_model_elements.vehicles.airplane import Airplane

from sim_model_elements.timer import Timer

from pydsol.model.basic_logger import get_module_logger

import logging

logger = get_module_logger(__name__, level=logging.INFO)


class ComplexSimModelGraph(DSOLModel):

    def __init__(self, simulator, input_params, graph, **kwargs):
        super().__init__(simulator, **kwargs)
        self.input_params = input_params
        self.seed = kwargs["seed"] if "seed" in kwargs else 1

        self.G = graph
        self.components = []
        self.links = []
        self.components_by_type = {}
        self.suppliers = None
        self.retailers = None
        self.endcustomer = None

    def construct_model_nodes(self):
        """Function that construct node simulation objects.
        """
        self.components = []
        for node, attributes in self.G.nodes(data=True):
            node_type = attributes['node_type']

            # restructure attributes
            for key, value in attributes.items():
                try:
                    attributes[key] = eval(value)
                except (TypeError, NameError, SyntaxError):
                    attributes[key] = value

            attributes["node_name"] = node
            attributes["seed"] = self.seed

            # make components
            if node_type == "supplier":
                component = Supplier(self.simulator, **attributes)

            elif node_type == "manufacturer":
                component = Manufacturer(self.simulator, **attributes)

            elif node_type == "wholesales_consolidator":
                component = WholesalesConsolidator(self.simulator, **attributes)

            elif node_type == "export_port":
                if attributes["modality"] == "sea":
                    component = Port(self.simulator, **attributes)
                elif attributes["modality"] == "air":
                    component = Airport(self.simulator, **attributes)

            elif node_type == "transit_port":
                if attributes["modality"] == "sea":
                    component = Port(self.simulator, **attributes)
                elif attributes["modality"] == "air":
                    component = Airport(self.simulator, **attributes)

            elif node_type == "import_port":
                if attributes["modality"] == "sea":
                    component = ImportPort(self.simulator, **attributes)
                elif attributes["modality"] == "air":
                    component = ImportAirport(self.simulator, **attributes)

            elif node_type == "wholesales_distributor":
                component = WholesalesDistributor(self.simulator, **attributes)

            elif node_type == "large_retailer":
                component = Retailer(self.simulator, **attributes)

            elif node_type == "small_retailer":
                component = Retailer(self.simulator, **attributes)

            elif node_type == "end_customer":
                component = EndCustomer(self.simulator)
                self.endcustomer = component

            else:
                raise ValueError("{0} had no node type".format(node))

            # to ensure the attributes are at the right level
            for key, value in attributes.items():
                setattr(component, key, value)
            self.components.append(component)

    def construct_model_links(self):
        """
        Constructs links between all entities.
        """
        for edge in self.G.edges(data=True):
            node1 = next((x for x in self.components if x.node_name == edge[0]), None)
            node2 = next((x for x in self.components if x.node_name == edge[1]), None)

            # for intermodal modalities
            if "other_modality" in edge[2]:
                other_modality = list(set(edge[2]["other_modality"]))
                dist_dict = edge[2]["distribution"] if "distribution" in edge[2] else {}
                if "deterministic" in dist_dict:
                    time_minutes = dist_dict["deterministic"]["params"][0]
                    self.links.append(IntermodalTimeLink(self.simulator, node1, node2, time_minutes, selection_weight=1,
                                                         other_modality=other_modality))
                elif "discrete" in dist_dict:
                    time_minutes = dist_dict["discrete"]["params"]
                    dist = dist_dict["discrete"]["model"]
                    self.links.append(IntermodalTimeLink(self.simulator, node1, node2, time_minutes, selection_weight=1,
                                                         distribution=dist, other_modality=other_modality))

                elif "dist" in dist_dict:
                    time_object = dist_dict["dist"]["model"]
                    self.links.append(IntermodalTimeLink(self.simulator, node1, node2, time_object, selection_weight=1,
                                                         other_modality=other_modality))

                elif "median_time_minutes" in edge[2]:
                    time_minutes = edge[2]["median_time_minutes"]
                    self.links.append(IntermodalTimeLink(self.simulator, node1, node2, time_minutes, selection_weight=1,
                                                         other_modality=other_modality))

            # for sea ports
            elif "port" in node1.node_type and "port" in node2.node_type and edge[2]["modality"] == "sea":
                dist_dict = edge[2]["distribution"] if "distribution" in edge[2] else {}
                if "deterministic" in dist_dict:
                    time_minutes = dist_dict["deterministic"]["params"][0]
                    self.links.append(SeaTimeLink(self.simulator, node1, node2, time_minutes, selection_weight=1))
                elif "discrete" in dist_dict:
                    time_minutes = dist_dict["discrete"]["params"]
                    dist = dist_dict["discrete"]["model"]
                    self.links.append(SeaTimeLink(self.simulator, node1, node2, time_minutes, selection_weight=1,
                                                  distribution=dist))

                elif "dist" in dist_dict:
                    time_object = dist_dict["dist"]["model"]
                    self.links.append(SeaTimeLink(self.simulator, node1, node2, time_object, selection_weight=1))

                elif "median_time_minutes" in edge[2]:
                    time_minutes = edge[2]["median_time_minutes"]
                    self.links.append(SeaTimeLink(self.simulator, node1, node2, time_minutes, selection_weight=1))

                # time_minutes = edge[2]["median_time_minutes"]
                # self.links.append(SeaTimeLink(self.simulator, node1, node2, time_minutes, selection_weight=1))

            # for flights and land routes
            else:
                distance = edge[2]["distance_km"]
                self.links.append(DistanceLink(self.simulator, node1, node2, distance, selection_weight=1))

        # Set up next connections
        for node, data in self.G.nodes(data=True):
            component = next((x for x in self.components if x.node_name == node), None)
            edges_of_node = [l for l in self.links if l.origin == component]
            component.next = edges_of_node if len(edges_of_node) > 0 else None
            for edge in edges_of_node:
                edge.next = edge.destination

    def construct_model(self):
        """Model has the time unit km/h"""
        # Reset model
        self.reset_model()
        np.random.seed(self.seed)

        # construct model
        self.construct_model_nodes()
        self.construct_model_links()

        # set output stats (minus end customer)
        self.components = [c for c in self.components if c.node_type != "end_customer"]
        self.set_output_stats()

    @staticmethod
    def reset_model():
        classes = [Product, Supplier, Manufacturer, WholesalesConsolidator,
                   Port, ImportPort, Airport, ImportAirport, WholesalesDistributor, Retailer,
                   Link, Feeder, Vessel, Airplane, SmallTruck,
                   LargeTruck, EndCustomer, Container, RawMaterials, Batch, Parcel]

        for i in classes:
            i.id_iter = itertools.count(1)

    def set_output_stats(self):
        timer = Timer(self.simulator, 1)
        for component in self.components:
            timer.set_event(component, "get_daily_stats")
            timer.set_event(component, "get_arrival_stats")

    def get_output_statistics(self):
        if self.retailers == None:
            self.retailers = [x for x in self.components if isinstance(x, Retailer)]
        if self.suppliers == None:
            self.suppliers = [x for x in self.components if isinstance(x, Supplier)]

        import_ports = [x for x in self.components if isinstance(x, ImportPort) or isinstance(x, ImportAirport)]

        # Detection Probability per Port
        detection_probability_per_port_relatively = {port.name:
                                                         (port.number_of_criminal_containers_checks / port.total_number_of_custom_checks)
                                                         if port.total_number_of_custom_checks != 0 else 0
                                                         for port in import_ports}

        detection_probability_per_port_containers = {port.name: (port.number_of_criminal_containers_checks / port.total_number_of_containers)
                                                     if port.total_number_of_containers != 0 else 0 for port in import_ports}
        detection_probability_per_port_parcels = {port.name: (port.seized_parcels / port.total_parcels)
                                                  if port.total_parcels != 0 else 0 for port in import_ports}

        # Detection Probability Total
        total_custom_checks = sum(map(lambda port: port.total_number_of_custom_checks, import_ports))
        detection_probability_relatively = (
            sum(map(lambda port: port.number_of_criminal_containers_checks, import_ports)) /
            total_custom_checks if total_custom_checks != 0 else 0)

        total_containers = sum(map(lambda port: port.total_number_of_containers, import_ports))
        detection_probability_containers = (
            sum(map(lambda port: port.number_of_criminal_containers_checks, import_ports)) /
            total_containers if total_containers != 0 else 0)

        total_parcels = sum(map(lambda port: port.total_parcels, import_ports))
        detection_probability_parcels = (
            sum(map(lambda port: port.seized_parcels, import_ports)) /
            total_parcels if total_parcels != 0 else 0)

        # Avg times
        average_product_time_in_system = [product.time_in_system for product in self.endcustomer.entities_of_system]
        average_supply_side_time = [product.supply_side_time for product in self.endcustomer.entities_of_system]
        average_international_transport_time = [product.international_transport_time for product in
                                                self.endcustomer.entities_of_system]
        average_wholesales_distributor_time = [product.wholesales_distributor_time for product in
                                               self.endcustomer.entities_of_system]
        average_demand_side_time = [product.demand_side_time for product in self.endcustomer.entities_of_system]

        # calculate average transport cost
        average_transport_cost = []
        for product in self.endcustomer.entities_of_system:
            total_travel_cost = sum([v for k, v in product.travel_cost.items()])
            multiply_factor = self.multiply_function_time_discount(product.time_in_system)
            travel_cost_product = total_travel_cost * multiply_factor
            average_transport_cost.append(travel_cost_product)

        average_quantity = []
        for supplier in self.suppliers:
            average_quantity += supplier.list_quantity

        outcomes = {"Time_In_System": statistics.mean(average_product_time_in_system) if len(
            average_product_time_in_system) > 0 else 0,
                    "Supply_Side_Time": statistics.mean(average_supply_side_time) if len(
                        average_supply_side_time) > 0 else 0,
                    "International_Transport_Time": statistics.mean(average_international_transport_time) if len(
                        average_international_transport_time) > 0 else 0,
                    "Wholesales_Time": statistics.mean(average_wholesales_distributor_time) if len(
                        average_wholesales_distributor_time) > 0 else 0,
                    "Demand_Side_Time": statistics.mean(average_demand_side_time) if len(
                        average_demand_side_time) > 0 else 0,
                    "Quantity": statistics.mean(average_quantity) if len(average_quantity) > 0 else 0,
                    "Transport_Cost": statistics.mean(average_transport_cost) if len(
                        average_transport_cost) > 0 else 0,
                    "Detection_Prob_Checks": detection_probability_relatively,
                    "Detection_Prob_Containers": detection_probability_containers,
                    "Detection_Prob_Parcels": detection_probability_parcels}

        # Save time series data
        time_series_dict = {}
        total_event_time_series = {}

        for idx, component in enumerate(self.components):
            key = component.name
            time_series_dict[key] = pd.Series(component.daily_stats)

            # Event based
            event_time_series = {}
            for i, (k, v) in enumerate(component.arrival_stats.items()):
                other = {"location": component.location, "type": component.node_type}
                v.update(other)
                event_time_series[str(idx) + str(i)] = v

            total_event_time_series.update(event_time_series)

        # Per day
        time_series = pd.DataFrame(time_series_dict)
        time_series.reset_index(inplace=True)
        time_series = time_series.rename(columns={"index": "Time"})

        # Event based
        event_time_series = pd.DataFrame.from_dict(total_event_time_series, orient="index")

        result = {"outcomes": outcomes, "time_series": event_time_series}

        return result

    def multiply_function_time_discount(self, total_time_in_system):
        v = (-0.34 / 730) * total_time_in_system + 0.59
        v_0 = 0.59

        # new - old/old
        pp = 1 - ((v - v_0) / (v_0))

        return pp
