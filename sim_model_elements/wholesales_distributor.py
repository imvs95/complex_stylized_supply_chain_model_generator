"""
Created on: 13-8-2021 10:48

@author: IvS
"""
import copy

import numpy as np

from sim_model_elements.entities import Product, Container
from pydsol.model.server import Server

from sim_model_elements.statistic_manager import StatisticManager

import logging
from pydsol.model.basic_logger import get_module_logger

logger = get_module_logger(__name__, level=logging.INFO)

class WholesalesDistributor(Server, StatisticManager):
    def __init__(self, simulator, processing_time, distribution=np.random.triangular,
                 **kwargs):
        super().__init__(simulator, distribution=distribution,
                         processing_time=processing_time, **kwargs)
        StatisticManager.__init__(self)
        self.location = kwargs['location']
        self.kwargs = kwargs

        # Override function processing - Resource
        self.processing_function_resource = self.resources[0].processing
        for resource in self.resources:
            resource.processing = self.processing

    def seize_resource(self, entity: Product, **kwargs):
        if isinstance(entity, Container):
            for product in entity.criminal_products_in_container:
                product.start_wholesales_distributor = self.simulator.simulator_time
                # product.international_transport_time = product.start_wholesales_distributor - product.start_international_transport

                product.route.append(self)
                product.travel_cost.update(entity.travel_cost)
                # product.on_container_international_transport = entity.name
                self.quantity += product.quantity
                self.arrivals_amount[product.name] = product.quantity

                super().seize_resource(product, **kwargs)
            del entity

        else:
            entity.start_wholesales_distributor = self.simulator.simulator_time
            # entity.international_transport_time = entity.start_wholesales_distributor - entity.start_international_transport

            entity.route.append(self)
            self.quantity += entity.quantity
            self.arrivals_amount[entity.name] = entity.quantity

            super().seize_resource(entity, **kwargs)

    def enter_resource(self):
        """Schedules the event to transfer into the resource and starts processing.

        Parameters
        ----------
        entity: object
            the target on which a state change is scheduled.
        """
        self.simulator.schedule_event_rel(self.transfer_in_time, self, "processing")

    def processing(self, **kwargs):
        #this only works if you have one resource
        for resource in self.resources:
            entity = resource.processing_entity
        if len(self.next) > 1:
            divide_quantity = len(self.next)
            batches_product = self.divide_entity(entity, divide_quantity)
            self.processing_function_resource(processing_entity=batches_product)
        elif len(self.next) == 1:
            self.processing_function_resource(processing_entity=entity)
            #for batch in batches_product:


    @staticmethod
    def divide_entity(entity: Product, copy_quantity: int):
        copies = []
        for i in range(copy_quantity):
            copy_entity = copy.copy(entity)
            copy_entity.quantity = entity.quantity / copy_quantity
            copy_entity.name = entity.name + "." + str(i)
            copy_entity.travel_cost = {k: v for k, v in entity.travel_cost.items()}
            copy_entity.route = [e for e in entity.route]
            copies.append(copy_entity)
        return copies

    def get_daily_stats(self):
        self.quantity = max(0, self.quantity)
        if self.quantity < 1:
            self.quantity = 0
        super().get_daily_stats()

    def enter_output_node(self, entity, **kwargs):
        #list of products due to dividing
        try:
            self.products_left.append(entity[0].name.split(".")[0])

            for batch in entity:
                self.quantity -= batch.quantity
                batch.wholesales_distributor_time = self.simulator.simulator_time - batch.start_wholesales_distributor

                super().enter_output_node(batch, **kwargs)

        except TypeError:
            #for only one next process
            self.products_left.append(entity.name)
            self.quantity -= entity.quantity
            entity.wholesales_distributor_time = self.simulator.simulator_time - entity.start_wholesales_distributor

            super().enter_output_node(entity, **kwargs)



    def exit_output_node(self, entity, **kwargs):
        #Add interarrival delay
        interarrival_delay = entity.interarrival_distribution(*entity.interarrival_times)
        self.simulator.schedule_event_rel(interarrival_delay, self, "exit_with_vehicle", entity=entity)

    def exit_with_vehicle(self, entity, **kwargs):
        logging.debug(
            "Time {0:.2f}: {1} exits {2}".format(self.simulator.simulator_time, entity.name, self.name))
        super().exit_output_node(entity, **kwargs)