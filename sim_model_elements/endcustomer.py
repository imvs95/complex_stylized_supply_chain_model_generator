"""
Created on: 5-4-2023 14:31

@author: IvS
"""

import numpy as np

from pydsol.model.sink import Sink

from sim_model_elements.statistic_manager import StatisticManager


class EndCustomer(Sink):
    def __init__(self, simulator, transfer_in_time=0, **kwargs):
        super().__init__(simulator, transfer_in_time=transfer_in_time
                         , **kwargs)
        self.name = "End Customer/Export"

        self.entities_of_system = []

    def enter_input_node(self, entity):
        super().enter_input_node(entity)

    def destroy_entity(self, entity, **kwargs):
        self.entities_of_system.append(entity)
        super().destroy_entity(entity)
