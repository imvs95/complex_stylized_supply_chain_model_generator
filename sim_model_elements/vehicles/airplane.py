import numpy as np
import itertools

from pydsol.model.entities import Vehicle


class Airplane(Vehicle):
    id_iter = itertools.count(1)

    def __init__(self, simulator, **kwargs):
        self.speed = np.random.uniform(740, 930) * 24  # km/h to day
        if "vehicle_speed" in kwargs:
            self.speed = np.random.triangular(740, kwargs["vehicle_speed"], 930) * 24

        super().__init__(simulator, self.speed, **kwargs)
        # self.boat_number = str()
        # self.containers_on_boat = []

        self.interarrival_distribution = np.random.triangular
        self.interarrival_times = (0, 1 / 24, 4 / 24)

        self.cost_per_time_unit = 15.7  # $/kg/day
        self.start_link = 0
        self.end_link = 0
        self.travel_time = 0
        self.travel_cost = 0

        self.id = next(self.id_iter)
        self.name = "{0} {1}".format(self.__class__.__name__, str(self.id))
        if "name" in kwargs:
            self.name = kwargs["name"]
