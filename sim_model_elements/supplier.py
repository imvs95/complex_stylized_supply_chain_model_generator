from sim_model_elements.entities import Product
from pydsol.model.source import Source
from sim_model_elements.statistic_manager import StatisticManager

import logging
import numpy as np
import copy

from pydsol.model.basic_logger import get_module_logger
import logging

# logging.basicConfig(level=logging.INFO,
#         format='%(asctime)s [%(levelname)s] %(message)s (%(name)s - %(filename)s: line %(lineno)s)')

logger = get_module_logger(__name__, level=logging.CRITICAL)

class Supplier(Source, StatisticManager):
    def __init__(self, simulator, interarrival_time, **kwargs):
        super().__init__(simulator, interarrival_time, **kwargs)
        StatisticManager.__init__(self)
        self.location = kwargs['location']
        self.list_quantity = []

    def create_entities(self, **kwargs):
        """
        Create entities via SimEvent, given the interarrival time and the number of entities.

        Parameters
        ----------
        entity_type: class
            class where to make instances of, for example class Entity
        kwargs:
            kwargs are the keyword arguments that are used to invoke the method or expand the function.

        """

        # To make it work with the set seed of the simulator
        if self.interarrival_time == "default":
            self.interarrival_time = np.random.exponential(0.25)

        # Create new entity
        for _ in range(self.num_entities):
            entity = self.entity_type(self.simulator, self.simulator.simulator_time, interarrival_time=self.interarrival_time)
            logging.debug("Time {0:.2f}: {1} is created at {2}".format(self.simulator.simulator_time, entity.name,
                                                                     self.name))

            self.exit_source(entity)

        interarrival_time = self.distribution(
            self.interarrival_time) if "distribution" in self.__dict__ else self.interarrival_time
        relative_delay = interarrival_time

        # Schedule event to create next entity according to the interarrival time
        self.simulator.schedule_event_rel(relative_delay, self, "create_entities")

    def exit_source(self, entity: Product, **kwargs):
        # Tally quantity
        self.list_quantity.append(entity.quantity)
        self.quantity += entity.quantity
        self.arrivals_amount[entity.name] = entity.quantity
        super().exit_source(entity, **kwargs)

    def enter_output_node(self, entity: Product, **kwargs):
        entity.route.append(self)
        self.quantity -= entity.quantity
        self.products_left.append(entity.name)

        # if there is more than one manufacturer, then divide it (equally) over all the manufacturers
        if isinstance(self.next, list) and len(self.next) > 1:
            divide_quantity = len(self.next)
            batches_product = self.divide_entity(entity, divide_quantity)
            for batch in batches_product:
                super().enter_output_node(batch, **kwargs)

        else:
            super().enter_output_node(entity, **kwargs)

    def exit_output_node(self, entity, **kwargs):
        #Add interarrival delay
        interarrival_delay = entity.interarrival_distribution(*entity.interarrival_times)

        self.simulator.schedule_event_rel(interarrival_delay, self, "exit_with_vehicle", entity=entity)

    def exit_with_vehicle(self, entity, **kwargs):
        super().exit_output_node(entity, **kwargs)

    @staticmethod
    def divide_entity(entity, copy_quantity: int):
        # we now divide it equally over all the manufacturers
        copies = []
        for i in range(copy_quantity):
            copy_entity = copy.copy(entity)
            copy_entity.quantity = entity.quantity / copy_quantity
            copy_entity.name = entity.name + "." + str(i)
            copy_entity.route = [i for i in entity.route]
            copy_entity.travel_cost = {}
            copies.append(copy_entity)
        return copies
