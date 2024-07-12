"""
Created on: 17-8-2021 13:18

@author: IvS
"""
import numpy as np

class Timer(object):
    def __init__(self, simulator, time, **kwargs):
        self.simulator = simulator
        self.time = time
        self.distribution = kwargs["distribution"] if "distribution" in kwargs else None

    def set_event(self, source_stats, method_stats, **kwargs):
        self.simulator.schedule_event_now(source_stats, method_stats, **kwargs)

        # new event for statistics one hour later
        self.simulator.schedule_event_rel(self.time, self, "set_event", source_stats=source_stats,
                                                   method_stats=method_stats, **kwargs)

    def set_event_dist(self, source_stats, method_stats, **kwargs):
        self.simulator.schedule_event_now(source_stats, method_stats, **kwargs)

        rel_time = self.distribution(*self.time)

        # new event for statistics one hour later
        self.simulator.schedule_event_rel(rel_time, self, "set_event_dist", source_stats=source_stats,
                                                   method_stats=method_stats, **kwargs)
