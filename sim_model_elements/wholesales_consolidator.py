"""
Created on: 13-8-2021 10:48

@author: IvS
"""
import copy

import numpy as np

from sim_model_elements.entities import Product, Container
from sim_model_elements.timer import Timer
from pydsol.model.server import Server

from sim_model_elements.statistic_manager import StatisticManager


class WholesalesConsolidator(Server, StatisticManager):

    def __init__(self, simulator, processing_time, distribution=np.random.triangular,
                 **kwargs):
        super().__init__(simulator, distribution=distribution,
                         processing_time=processing_time, **kwargs)
        StatisticManager.__init__(self)
        self.location = kwargs['location']
        self.kwargs = kwargs

        self.parcels_at_consolidator = []

        # for order pickup
        if "wc_pickuptime" in self.kwargs:
            timer = Timer(self.simulator, self.kwargs["wc_pickuptime"], distribution=np.random.triangular)
        else:
            timer = Timer(self.simulator, (0.5, 1, 2), distribution=np.random.triangular)

        timer.set_event_dist(self, "arrive_pickup_person")

    def seize_resource(self, entity, **kwargs):
        entity.route.append(self)
        self.quantity += entity.quantity
        self.arrivals_amount[entity.name] = entity.quantity

        super().seize_resource(entity, **kwargs)

    def enter_output_node(self, entity, **kwargs):
        for parcel in entity.parcels:
            self.parcels_at_consolidator.append(parcel)

        # wait until pick up
        time_until_pickup = np.random.triangular(0.5, 1, 2) if "wc_pickuptime" not in self.kwargs \
            else np.random.triangular(*self.kwargs["wc_pickuptime"])

        del entity

    def arrive_pickup_person(self, **kwargs):
        if self.simulator.simulator_time == 0:
            # if simulation time is 0
            pass
        elif len(self.parcels_at_consolidator) == 0:
            # if no parcels at the consolidator
            pass
        else:
            # determine parcels to pick up and make Product instance
            if len(self.parcels_at_consolidator) == 1:
                num_parcels_pickup = 1
            elif len(self.parcels_at_consolidator) > 10:
                num_parcels_pickup = np.random.randint(10, len(self.parcels_at_consolidator))
            else:
                num_parcels_pickup = np.random.randint(1, len(self.parcels_at_consolidator))

            parcels_for_pickup = self.parcels_at_consolidator[:num_parcels_pickup]
            self.parcels_at_consolidator = [p for p in self.parcels_at_consolidator if p not in parcels_for_pickup]

            product = Product(self.simulator, self.simulator.simulator_time)
            product.parcels = parcels_for_pickup
            product.quantity = sum([p.number_of_units for p in parcels_for_pickup])
            product.start_time = min([p.start_time for p in parcels_for_pickup])
            product.manufacturer_time = min([p.manufacturer_time for p in parcels_for_pickup])
            product.route = parcels_for_pickup[0].route # pick first route -- however, make this maybe different? (of min start time)

            # get unique transport modes
            unique_travel_cost = list(set([(k, v) for p in parcels_for_pickup for k, v in p.travel_cost.items()]))
            product.travel_cost = {k: v for k, v in unique_travel_cost}

            for parcel in product.parcels:
                parcel.product_name = product.name

            self.quantity -= product.quantity
            self.products_left.append(product.name)

            transport_container = np.random.choice([True, False], p=[0.5, 0.5]) if "wc_prob_transport_container" not in self.kwargs else \
                np.random.choice([True, False], p=[self.kwargs["wc_prob_transport_container"], 1-self.kwargs["wc_prob_transport_container"]])
            # depending on the MO

            if transport_container == True:
                container = Container(self.simulator)
                container.criminal_products_in_container.append(product)
                product.on_container_start_location = self
                container.criminal = True
                product = container

            super().enter_output_node(product, **kwargs)

    def exit_output_node(self, entity, **kwargs):

        # Add that we can put the stuff already on the container before transport (or not!)
        # Add interarrival delay
        interarrival_delay = entity.interarrival_distribution(*entity.interarrival_times)

        self.simulator.schedule_event_rel(interarrival_delay, self, "exit_with_vehicle", entity=entity)

    def exit_with_vehicle(self, entity, **kwargs):
        super().exit_output_node(entity, **kwargs)

    def get_daily_stats(self):
        self.quantity = max(0, self.quantity)
        if self.quantity < 1:
            self.quantity = 0
        super().get_daily_stats()
