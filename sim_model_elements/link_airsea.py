"""
Created on: 14-2-2022 09:57

@author: IvS
"""
import numpy as np
import scipy.stats

from pydsol.model.link import Link, TimePath
from sim_model_elements.port import ImportPort

from pydsol.model.basic_logger import get_module_logger

logger = get_module_logger(__name__)

class SeaTimeLink(TimePath):
    """This class defines a link at sea with time in days -- input is time in minutes."""

    def __init__(self, simulator, origin, destination, time, **kwargs):
        if isinstance(time, (int, float, np.integer, np.floating)):
            time_days = time / 60 / 24
        else:
            time_days = time

        super().__init__(simulator, origin, destination, time=time_days, **kwargs)
        self.distribution = kwargs["distribution"] if "distribution" in kwargs else None

    def enter_input_node(self, entity, **kwargs):
        entity.start_link = self.simulator.simulator_time
        super().enter_input_node(entity)

    def enter_path(self, entity, **kwargs):
        if isinstance(self.time, scipy.stats._distn_infrastructure.rv_continuous_frozen):
            # When distribution fitted
            relative_delay = self.time.rvs(random_state=self.simulator.seed) / 60 / 24
            if relative_delay < 0:
                self.time.args = tuple(abs(x) for x in self.time.args)
                relative_delay = self.time.rvs(random_state=self.simulator.seed) / 60 / 24

        elif isinstance(self.distribution, type(np.random.RandomState().choice)):
            # When discrete distribution
            relative_delay = np.random.choice(self.time[0], p=self.time[1]) / 60 / 24

        else:
            try:
                relative_delay = self.distribution(*self.time) / 60 / 24
            except TypeError:
                relative_delay = self.time

        if isinstance(self.destination, ImportPort):
            # only happens at Import Port at Sea
            # time when they submit the SAL which determines whether there is a check or not
            # it has to be submitted 72-0 hours before eta
            for container in entity.entities_on_vehicle:
                time_SAL = np.random.randint(0.1, 72) / 24
                eta = relative_delay
                self.simulator.schedule_event_rel(max(0, (eta - time_SAL)), self.destination, "determine_custom_check",
                                                  entity=container)

        self.simulator.schedule_event_rel(relative_delay, self, "exit_path", entity=entity)

        logger.info("Time {0:.2f}: {1} enters {2} from {3} to {4}".format(self.simulator.simulator_time,
                                                                          entity.name, self.name,
                                                                          self.origin.name, self.destination.name))

    def exit_path(self, entity, **kwargs):
        entity.end_link = self.simulator.simulator_time
        entity.travel_time = entity.end_link - entity.start_link
        volume_entity = sum([par.volume for c in entity.entities_on_vehicle for p in c.criminal_products_in_container for
                         par in p.parcels])
        entity.travel_cost = entity.travel_time * entity.cost_per_time_unit * volume_entity


        for p in entity.entities_on_vehicle:
            p.travel_cost[entity.name] = entity.travel_cost

        super().exit_path(entity)

class SeaLink(Link):
    """This class defines a link at sea with distance"""

    def __init__(self, simulator, origin, destination, length, **kwargs):
        super().__init__(simulator, origin, destination, length, **kwargs)

    def enter_input_node(self, entity, **kwargs):
        entity.start_link = self.simulator.simulator_time
        super().enter_input_node(entity)

    def enter_link(self, entity, **kwargs):
        distance_travelled = 0
        days_to_travel = 0

        while distance_travelled < self.length:
            distance_per_day = entity.speed_distribution(*entity.speed_values) * 24

            distance_to_travel = self.length - distance_travelled
            if distance_per_day > distance_to_travel:
                distance_travelled += distance_to_travel
                days_to_travel += max(0, (distance_to_travel / distance_per_day))
            else:
                distance_travelled += distance_per_day
                days_to_travel += 1

        relative_delay = days_to_travel  # per day

        self.simulator.schedule_event_rel(relative_delay, self, "exit_link", entity=entity)

    def exit_link(self, entity, **kwargs):
        entity.end_link = self.simulator.simulator_time
        entity.travel_time = entity.end_link - entity.start_link
        volume_entity = sum([par.volume for c in entity.entities_on_vehicle for p in c.criminal_products_in_container for
                         par in p.parcels])
        entity.travel_cost = entity.travel_time * entity.cost_per_time_unit * volume_entity

        for p in entity.entities_on_vehicle:
            p.travel_cost[entity.name] = entity.travel_cost

        super().exit_link(entity)
