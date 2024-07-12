"""
Created on: 16-1-2023 15:48

@author: IvS
"""

import itertools
import statistics
import numpy as np
import pandas as pd
import time

from pydsol.core.model import DSOLModel

from sim_model_elements.entities import RawMaterials, Batch, Parcel, Product, Container
from sim_model_elements.supplier import Supplier
from sim_model_elements.manufacturer import Manufacturer
from sim_model_elements.wholesales_consolidator import WholesalesConsolidator
from sim_model_elements.port import Port, ImportPort
from sim_model_elements.wholesales_distributor import WholesalesDistributor
from sim_model_elements.retailer import Retailer
from sim_model_elements.endcustomer import EndCustomer
from pydsol.model.link import Link
from sim_model_elements.link_airsea import SeaLink, SeaTimeLink

from sim_model_elements.vehicles.large_truck import LargeTruck
from sim_model_elements.vehicles.small_truck import SmallTruck
from sim_model_elements.vehicles.feeder import Feeder
from sim_model_elements.vehicles.vessel import Vessel

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
        for node, attributes in self.G.nodes(data=True):
            node_type = attributes['node_type']

            # restructure attributes
            for key, value in attributes.items():
                try:
                    attributes[key] = eval(value)
                except (TypeError, NameError, SyntaxError):
                    attributes[key] = value

            # make components
            if node_type == "supplier":
                component = Supplier(self.simulator, **attributes)

            if node_type == "manufacturer":
                component = Manufacturer(self.simulator, **attributes)

            if node_type == "wholesales_consolidator":
                component = WholesalesConsolidator(self.simulator, **attributes)

            if node_type == "export_port":
                component = Port(self.simulator, **attributes)

            if node_type == "transit_port":
                component = Port(self.simulator, **attributes)

            if node_type == "import_port":
                component = ImportPort(self.simulator, **attributes)

            if node_type == "wholesales_distributor":
                component = WholesalesDistributor(self.simulator, **attributes)

            if node_type == "large_retailer":
                component = Retailer(self.simulator, **attributes)

            if node_type == "small_retailer":
                component = Retailer(self.simulator, **attributes)

            # to ensure the attributes are at the right level
            for key, value in attributes.items():
                setattr(component, key, value)
            self.components.append(component)

        # Add the final customer
        self.endcustomer = EndCustomer(self.simulator)

    def construct_model_links(self):
        """
        Constructs links between all entities.
        """
        for edge in self.G.edges(data=True):
            node1 = next((x for x in self.components if x.node_type + "_" + str(x.n) == edge[0]), None)
            node2 = next((x for x in self.components if x.node_type + "_" + str(x.n) == edge[1]), None)

            if "port" in node1.node_type and "port" in node2.node_type:
                time_minutes = edge[2]["distance_km"]
                self.links.append(SeaTimeLink(self.simulator, node1, node2, time_minutes, selection_weight=1))
            else:
                distance = edge[2]["distance_km"]
                self.links.append(Link(self.simulator, node1, node2, distance, selection_weight=1))

        # Set up next connections
        for node, data in self.G.nodes(data=True):
            component = next((x for x in self.components if x.node_type + "_" + str(x.n) == node), None)
            edges_of_node = [l for l in self.links if l.origin == component]
            component.next = edges_of_node if len(edges_of_node) > 0 else None
            for edge in edges_of_node:
                edge.next = edge.destination

        # Set up all connections from retailer to end customer
        for comp in self.components:
            if comp.node_type == "small_retailer":
                comp.next = self.endcustomer

    def construct_model(self):
        """Model has the time unit km/h"""
        # Reset model
        self.reset_model()
        np.random.seed(self.seed)

        # construct model
        self.construct_model_nodes()
        self.construct_model_links()
        self.set_output_stats()

    @staticmethod
    def reset_model():
        classes = [Product, Supplier, Manufacturer, WholesalesConsolidator,
                   Port, WholesalesDistributor, Retailer,
                   Link, Feeder, Vessel, SmallTruck,
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

        import_ports = [x for x in self.components if isinstance(x, ImportPort)]
        detection_probability_per_port = {port.name: port.detection_probability for port in import_ports}

        average_product_time_in_system = [product.time_in_system for product in self.endcustomer.entities_of_system]
        average_supply_side_time = [product.supply_side_time for product in self.endcustomer.entities_of_system]
        average_international_transport_time = [product.international_transport_time for product in
                                                self.endcustomer.entities_of_system]
        average_wholesales_distributor_time = [product.wholesales_distributor_time for product in
                                               self.endcustomer.entities_of_system]
        average_demand_side_time = [product.demand_side_time for product in self.endcustomer.entities_of_system]

        average_quantity = []
        for supplier in self.suppliers:
            average_quantity += supplier.list_quantity

        outcomes = {"Time_In_System": statistics.mean(average_product_time_in_system),
                    "Supply_Side_Time": statistics.mean(average_supply_side_time),
                    "International_Transport_Time": statistics.mean(average_international_transport_time),
                    "Wholesales_Time": statistics.mean(average_wholesales_distributor_time),
                    "Demand_Side_Time": statistics.mean(average_demand_side_time),
                    "Quantity": statistics.mean(average_quantity)}

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

        result = {"outcomes": outcomes, "time_series": event_time_series, "graph": self.G}

        return result
