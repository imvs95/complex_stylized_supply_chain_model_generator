import numpy as np

from pydsol.model.entities import Entity
import copy


class RawMaterials(Entity):
    def __init__(self, simulator, t, **kwargs):
        super().__init__(simulator, t, **kwargs)

        self.raw_materials_quantity = self.calculate_raw_materials_quantity(self.kwargs["interarrival_time"])
        self.quantity = self.raw_materials_quantity
        self.start_time = self.simulator.simulator_time
        self.start_manufacturer = 0

        self.location = str
        self.ha = int

        self.route = []

        self.travel_cost = {}

    def __repr__(self):
        return self.name


    def calculate_raw_materials_quantity(self, interarrival_time):
        q = round(np.random.uniform(10000, 40000) * interarrival_time, 0)
        return q


class Batch(Entity):
    def __init__(self, simulator, t, **kwargs):
        super().__init__(simulator, t, **kwargs)

        self.start_time = self.simulator.simulator_time
        self.time_in_system = 0
        self.start_international_transport = 0
        self.start_manufacturer = 0

        self.supply_side_time = 0
        self.manufacturer_time = 0

        self.route = []

        self.parcels = []
        self.total_parcels = int
        self.total_units = int
        self.quantity = int

        self.travel_cost = {}

    def __repr__(self):
        return self.name

class Parcel(Entity):
    def __init__(self, simulator, t, **kwargs):
        super().__init__(simulator, t, **kwargs)

        self.raw_material_quantity = 0
        self.batch_name = 0
        self.product_name = 0
        self.number_of_units = 1
        self.volume = 0

        self.start_time = self.simulator.simulator_time
        self.time_in_system = 0
        self.start_international_transport = 0
        self.start_wholesales_distributor = 0
        self.start_export_port = 0
        self.start_import_port = 0
        self.start_manufacturer = 0

        self.supply_side_time = 0
        self.manufacturer_time = 0
        self.international_transport_time = 0
        self.wholesales_distributor_time = 0
        self.demand_side_time = 0

        self.route = []
        self.on_container_international_transport = str
        self.on_container_start_location = object

        self.travel_cost = {}

    def __repr__(self):
        return self.name

class Product(Entity):
    def __init__(self, simulator, t, **kwargs):
        super().__init__(simulator, t, **kwargs)

        self.quantity = 0
        self.start_time = self.simulator.simulator_time
        self.time_in_system = 0
        self.start_international_transport = 0
        self.start_wholesales_distributor = 0
        self.start_export_port = 0
        self.start_import_port = 0
        self.start_manufacturer = 0

        self.supply_side_time = 0
        self.manufacturer_time = 0
        self.international_transport_time = 0
        self.wholesales_distributor_time = 0
        self.demand_side_time = 0

        self.parcels = []

        self.route = []
        self.shipping_route = []
        self.on_container_international_transport = str
        self.on_container_start_location = object

        self.bill_of_loading = 2 #containers
        self.custom_checks_import_port = bool
        self.place_terminal_import_port = (0, 0, 0) #row, column, height
        self.arrival_time_transport = 0
        self.extracting_import_port = bool

        self.next_link_ports_imp = object

        self.travel_cost = {}

    def __repr__(self):
        return self.name


class Container(Entity):
    def __init__(self, simulator, **kwargs):
        super().__init__(simulator=simulator, t=simulator.simulator_time, **kwargs)

        self.start_time = self.simulator.simulator_time
        self.time_in_system = 0
        self.start_international_transport = 0
        self.start_wholesales_distributor = 0
        self.start_export_port = 0
        self.start_import_port = 0
        self.start_manufacturer = 0

        self.supply_side_time = 0
        self.manufacturer_time = 0
        self.international_transport_time = 0
        self.wholesales_distributor_time = 0
        self.demand_side_time = 0

        self.shipping_route = []

        self.bill_of_loading = 2 #containers
        self.custom_checks_import_port = bool
        self.place_terminal_import_port = (0, 0, 0) #row, column, height
        self.arrival_time_transport = 0
        self.products_in_container = str

        self.next_link_ports_imp = object

        self.criminal_products_in_container = list()
        self.criminal = False

        self.travel_cost = {}

    def __repr__(self):
        return self.name

