import numpy as np

from sim_model_elements.entities import Product, Parcel, Batch
from pydsol.model.server import Server
from sim_model_elements.statistic_manager import StatisticManager


class Manufacturer(Server, StatisticManager):
    def __init__(self, simulator, processing_time, **kwargs):
        super().__init__(simulator, distribution=np.random.gamma,
                         processing_time=(processing_time, 0.5*processing_time), **kwargs)
        StatisticManager.__init__(self)

        self.location = kwargs['location']

    def seize_resource(self, entity: Product, **kwargs):
        entity.route.append(self)
        self.quantity += entity.quantity
        self.arrivals_amount[entity.name] = entity.quantity

        entity.start_manufacturer = self.simulator.simulator_time

        super().seize_resource(entity, **kwargs)

    def enter_output_node(self, entity, **kwargs):
        self.quantity -= entity.quantity
        self.products_left.append(entity.name)

        total_units_created = entity.quantity/10 #need 10 raw materials for one unit
        batch = Batch(self.simulator, self.simulator.simulator_time)
        units_per_parcel = 20
        weight_per_parcel = (units_per_parcel * 18)/1000  # kg
        num_parcels = round(total_units_created/units_per_parcel)
        for _ in range(num_parcels):
            parcel = Parcel(self.simulator, self.simulator.simulator_time)
            parcel.start_time = entity.start_time
            parcel.start_manufacturer = entity.start_manufacturer
            parcel.raw_material_quantity = entity.quantity
            parcel.route = entity.route
            parcel.batch_name = batch.name
            parcel.number_of_units = units_per_parcel
            parcel.volume = weight_per_parcel
            parcel.manufacturer_time = self.simulator.simulator_time - entity.start_manufacturer
            parcel.travel_cost = entity.travel_cost
            batch.parcels.append(parcel)

        batch.total_parcels = num_parcels
        batch.total_units = num_parcels*units_per_parcel
        batch.quantity = batch.total_units
        batch.start_time = entity.start_time
        batch.start_manufacturer = entity.start_manufacturer
        batch.manufacturer_time = self.simulator.simulator_time - batch.start_manufacturer
        batch.route = entity.route
        batch.travel_cost = entity.travel_cost

        del entity

        super().enter_output_node(batch, **kwargs)

    def exit_output_node(self, entity, **kwargs):
        #Add interarrival delay
        interarrival_delay = entity.interarrival_distribution(*entity.interarrival_times)

        self.simulator.schedule_event_rel(interarrival_delay, self, "exit_with_vehicle", entity=entity)

    def exit_with_vehicle(self, entity, **kwargs):
        super().exit_output_node(entity, **kwargs)
