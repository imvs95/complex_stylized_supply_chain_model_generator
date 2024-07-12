import numpy as np
import itertools

from pydsol.model.entities import Vehicle


class Feeder(Vehicle):
    id_iter = itertools.count(1)

    def __init__(self, simulator, **kwargs):
        # km/h to knots, and one knot is one nautical mile per hour
        self.speed = np.random.triangular(10 * 1.85, 18 * 1.85, 25 * 1.85)
        if "vehicle_speed" in kwargs:
            self.speed = np.random.triangular(10 * 1.85, kwargs["vehicle_speed"], 25 * 1.85)

        self.speed_distribution = np.random.triangular
        mode = kwargs["vehicle_speed"] if "vehicle_speed" in kwargs else 18*1.85
        self.speed_values = (10*1.85, mode, 25*1.85)

        #self.speed = None #to make it crash when it does not use sea link
        super().__init__(simulator, self.speed, **kwargs)
        # self.boat_number = str()
        # self.containers_on_boat = []

        self.interarrival_distribution = np.random.triangular
        self.interarrival_times = (0, 4, 16)

        self.cost_per_time_unit = 0.1  # $/kg/day
        self.start_link = 0
        self.end_link = 0
        self.travel_time = 0
        self.travel_cost = 0

        self.id = next(self.id_iter)
        self.name = "{0} {1}".format(self.__class__.__name__, str(self.id))
        if "name" in kwargs:
            self.name = kwargs["name"]