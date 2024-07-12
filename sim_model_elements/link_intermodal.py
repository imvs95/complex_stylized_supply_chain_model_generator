"""
Created on: 14-2-2022 09:57

@author: IvS
"""
import numpy as np
import scipy.stats

from pydsol.model.link import Link, TimePath
from pydsol.model.entities import Vehicle
from sim_model_elements.vehicles.airplane import Airplane

from pydsol.model.basic_logger import get_module_logger

logger = get_module_logger(__name__)


class IntermodalTimeLink(TimePath):
    """This class defines a link at sea with time in days -- input is time in minutes."""

    def __init__(self, simulator, origin, destination, time, **kwargs):
        if isinstance(time, (int, float, np.integer, np.floating)):
            time_days = time / 60 / 24
        else:
            time_days = time

        super().__init__(simulator, origin, destination, time=time_days, **kwargs)
        self.distribution = kwargs["distribution"] if "distribution" in kwargs else None
        self.modality = kwargs["other_modality"] if "other_modality" in kwargs else None

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

        self.simulator.schedule_event_rel(relative_delay, self, "exit_path", entity=entity)

        logger.info("Time {0:.2f}: {1} enters {2} from {3} to {4}".format(self.simulator.simulator_time,
                                                                          entity.name, self.name,
                                                                          self.origin.name, self.destination.name))

    def exit_path(self, entity, **kwargs):
        entity.end_link = self.simulator.simulator_time
        entity.travel_time = entity.end_link - entity.start_link
        entity.travel_cost = entity.travel_time * entity.cost_per_time_unit

        for p in entity.entities_on_vehicle:
            p.travel_cost[entity.name] = entity.travel_cost

        super().exit_path(entity)


class DistanceLink(Link):
    def __init__(self, simulator, origin, destination, length, selection_weight=1, **kwargs):
        super().__init__(simulator, origin, destination, length, selection_weight=selection_weight, **kwargs)

    def enter_input_node(self, entity, **kwargs):
        if isinstance(entity, Vehicle):
            entity.start_link = self.simulator.simulator_time
        else:
            pass
        super().enter_input_node(entity)

    def exit_link(self, entity, **kwargs):
        if isinstance(entity, Vehicle):
            entity.end_link = self.simulator.simulator_time
            entity.travel_time = entity.end_link - entity.start_link

            if isinstance(entity, Airplane):
                volume_entity = sum([par.volume for c in entity.entities_on_vehicle for p in
                                     c.criminal_products_in_container
                                     for
                                     par in p.parcels])
                entity.travel_cost = entity.travel_time * entity.cost_per_time_unit * volume_entity
            else:
                entity.travel_cost = entity.travel_time * entity.cost_per_time_unit

            for p in entity.entities_on_vehicle:
                p.travel_cost[entity.name] = entity.travel_cost
        else:
            pass

        super().exit_link(entity)
