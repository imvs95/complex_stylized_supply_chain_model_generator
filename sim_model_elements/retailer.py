import logging
import numpy as np

from pydsol.model.entities import Vehicle
import copy
from pydsol.model.server import Server

from sim_model_elements.statistic_manager import StatisticManager


class Retailer(Server, StatisticManager):
    def __init__(self, simulator, processing_time, distribution=np.random.exponential, **kwargs):
        super().__init__(simulator, processing_time=processing_time, distribution=distribution,
                         **kwargs)
        StatisticManager.__init__(self)

        self.location = kwargs['location']
        #self.name = self.name + " " + self.location

        self.entities_of_system = []


    def seize_resource(self, entity, **kwargs):
        entity.route.append(self)
        self.quantity += entity.quantity
        self.arrivals_amount[entity.name] = entity.quantity

        super().seize_resource(entity, **kwargs)

    def enter_output_node(self, entity, **kwargs):
        self.quantity -= entity.quantity
        self.products_left.append(entity.name)

        entity.time_in_system = self.simulator.simulator_time - entity.start_time
        entity.demand_side_time = self.simulator.simulator_time - entity.start_import_port

        self.entities_of_system.append(entity)

        #if there is more than one end customer, then divide it (equally) over all the retailers
        if isinstance(self.next, list) and len(self.next) > 1:
            divide_quantity = len(self.next)
            batches_product = self.divide_entity(entity, divide_quantity)
            for batch in batches_product:
                super().enter_output_node(batch, **kwargs)

        else:
            super().enter_output_node(entity, **kwargs)

    def exit_output_node(self, entity, **kwargs):
        if isinstance(entity, Vehicle):
            #Add interarrival delay
            interarrival_delay = entity.interarrival_distribution(*entity.interarrival_times)

            self.simulator.schedule_event_rel(interarrival_delay, self, "exit_with_vehicle", entity=entity)
        else:
            super().exit_output_node(entity, **kwargs)

    def exit_with_vehicle(self, entity, **kwargs):
        super().exit_output_node(entity, **kwargs)

    def get_daily_stats(self):
        self.quantity = max(0, self.quantity)
        if self.quantity < 1:
            self.quantity = 0
        super().get_daily_stats()

    @staticmethod
    def divide_entity(entity, copy_quantity: int):
        # we now divide it equally -- maybe also change this
        copies = []
        for i in range(copy_quantity):
            copy_entity = copy.copy(entity)
            copy_entity.quantity = entity.quantity / copy_quantity
            copy_entity.name = entity.name + "." + str(i)
            copy_entity.travel_cost = {k: v for k, v in entity.travel_cost.items()}
            copy_entity.route = [e for e in entity.route]
            copies.append(copy_entity)
        return copies
