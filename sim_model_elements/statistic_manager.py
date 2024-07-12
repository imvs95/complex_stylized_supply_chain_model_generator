"""
Created on: 17-8-2021 14:52

@author: IvS
"""


class StatisticManager(object):
    def __init__(self):
        self.quantity = 0
        self.daily_stats = {}

        self.products_left = []
        self.arrivals_amount = {}
        self.arrival_stats = {}

    def get_daily_stats(self):
        """" Statistic for the quantity after exactly one day. So at each 24 hours, how much
        quantity is present at that moment."""
        self.daily_stats[self.simulator.simulator_time] = self.quantity

    def get_arrival_stats(self):
        """Statistic for the quantity over the entire day. So in the last 24 hours, these products have arrived or
        are still present."""

        for k, v in self.arrivals_amount.items():
            self.arrival_stats[str(k) + str(self.simulator.simulator_time)] = \
                {"time": self.simulator.simulator_time, "quantity": v,
                 "item": k}

        self.arrivals_amount = {k: v for k, v in self.arrivals_amount.items() if k not in self.products_left}

        self.products_left = []



