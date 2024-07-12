import numpy as np

from pydsol.model.entities import Vehicle


class SmallTruck(Vehicle):
    def __init__(self, simulator, **kwargs):
        self.speed = np.random.triangular(30, 80, 120)*24 #km/h to day
        if "vehicle_speed" in kwargs:
            self.speed = np.random.triangular(0, kwargs["vehicle_speed"], 120)*24
        super().__init__(simulator, self.speed, **kwargs)

        self.interarrival_distribution = np.random.triangular
        self.interarrival_times = (0, 0.2, 0.5)


        self.cost_per_time_unit = 5.4  # $/day
        self.start_link = 0
        self.end_link = 0
        self.travel_time = 0
        self.travel_cost = 0