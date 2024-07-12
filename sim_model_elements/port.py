import numpy as np
import sys

from sim_model_elements.entities import Product, Container
from sim_model_elements.vehicles.vessel import Vessel
from sim_model_elements.vehicles.large_truck import LargeTruck
from sim_model_elements.vehicles.train import Train
from sim_model_elements.link_intermodal import IntermodalTimeLink

from pydsol.model.server import Server
from pydsol.model.resource import Resource
from pydsol.model.entities import Vehicle
from sim_model_elements.statistic_manager import StatisticManager

import logging
from pydsol.model.basic_logger import get_module_logger

#logging.basicConfig(level=logging.CRITICAL)
logger = get_module_logger(__name__, level=logging.CRITICAL)


class Port(Server, StatisticManager):
    def __init__(self, simulator, processing_time, capacity=1, distribution=np.random.triangular, **kwargs):
        # often only with transit - but we can use the regular processing resource
        if "processing_distribution" in kwargs:
            processing_dist_dict = kwargs["processing_distribution"]
            if "deterministic" in processing_dist_dict:
                self.processing_time = processing_dist_dict["deterministic"]["params"][0] / 60 / 24
                self.distribution = None

        super().__init__(simulator, capacity=capacity, distribution=distribution, processing_time=processing_time,
                         **kwargs)
        StatisticManager.__init__(self)
        self.location = kwargs['location']
        self.name = self.name + " " + self.location
        self.kwargs = kwargs

        # often only with transit - need to overwrite the resource (has to be written first)
        if "processing_distribution" in kwargs:
            processing_dist_dict = kwargs["processing_distribution"]
            if "discrete" in processing_dist_dict:
                time_params = processing_dist_dict["discrete"]["params"]
                dist = processing_dist_dict["discrete"]["model"]

                self.resources = []
                for i in range(1, self.capacity + 1):
                    resource = ResourceTransit(i, self.simulator, self.input_queue, distribution=dist,
                                               processing_time=time_params, transfer_in_time=self.transfer_in_time)
                    resource.server = self

                    self.resources.append(resource)

            elif "dist" in processing_dist_dict:
                self.processing_distribution_transit = processing_dist_dict["dist"]["model"]
                time_params = tuple(abs(x) for x in self.processing_distribution_transit.args)
                dist = self.processing_distribution_transit.dist

                self.resources = []
                for i in range(1, self.capacity + 1):
                    resource = ResourceTransit(i, self.simulator, self.input_queue, distribution=dist,
                                                   processing_time=time_params, transfer_in_time=self.transfer_in_time)
                    resource.server = self

                    self.resources.append(resource)

    def enter_input_node(self, entity, **kwargs):
        not_seized = [resource for resource in self.resources if not resource.resource_seized]
        new_transfer_in_time = np.random.triangular(*self.kwargs["transfer_in_time"]) if "transfer_in_time" in self.kwargs else 0

        for resource in not_seized:
            resource.transfer_in_time = new_transfer_in_time

        logging.debug("Time {0:.2f}: {1} loaded off at {2}".format(self.simulator.simulator_time, entity.name,
                                                              self.name))

        super().enter_input_node(entity, **kwargs)

    def seize_resource(self, entity: [Product, Container], **kwargs):
        if isinstance(entity, Container):
            entity.shipping_route.append(self)
            self.quantity += np.sum([products.quantity for products in entity.criminal_products_in_container])
            for product in entity.criminal_products_in_container:
                self.arrivals_amount[product.name] = product.quantity
                # product.shipping_route.append(self)
                # product.route.append(self)

        elif isinstance(entity, Product):
            self.quantity += entity.quantity
            self.arrivals_amount[entity.name] = entity.quantity

            container = Container(self.simulator)
            container.criminal_products_in_container.append(entity)
            container.shipping_route.append(self)
            entity.on_container_start_location = self
            container.criminal = True
            # for product in container.criminal_products_in_container:
            #     product.shipping_route.append(self)
            #     product.route.append(self)
            entity = container

        super().seize_resource(entity, **kwargs)

    def enter_output_node(self, entity: [Product, Container], **kwargs):
        if isinstance(entity, Container):
            self.quantity -= np.sum([products.quantity for products in entity.criminal_products_in_container])
            products_left = [products for products in entity.criminal_products_in_container]
            self.products_left.extend(products_left)
        else:
            self.quantity -= entity.quantity
            self.products_left.append(entity.name)

        if self.node_type == "export_port":
            if isinstance(entity, Product):
                entity.start_international_transport = self.simulator.simulator_time
                entity.supply_side_time = (self.simulator.simulator_time - entity.start_time)
            elif isinstance(entity, Container):
                for product in entity.criminal_products_in_container:
                    product.start_international_transport = self.simulator.simulator_time
                    product.supply_side_time = (self.simulator.simulator_time - product.start_time)

        logging.debug("Time {0:.2f}: {1} finished at {2}".format(self.simulator.simulator_time, entity.name,
                                                              self.name))

        next_link_port = self.choose_next_link(entity)
        if isinstance(next_link_port, IntermodalTimeLink):
            # if intermodal or rail
            if "RAIL" in next_link_port.modality:
                vehicle = Train(self.simulator)
            # if intermodal or truck
            elif "INTERMODAL" in next_link_port.modality or "TRUCK" in next_link_port.modality:
                vehicle = LargeTruck(self.simulator)
        # general case
        else:
            if "vehicle_type" in self.kwargs:
                if "vehicle_speed" in self.kwargs:
                    vehicle = self.kwargs["vehicle_type"](self.simulator, vehicle_speed=self.kwargs["vehicle_speed"])
                else:
                    vehicle = self.kwargs["vehicle_type"](self.simulator)

        vehicle.entities_on_vehicle.append(entity)
        vehicle.next_link_port = next_link_port
        logger.info("Time {0:.2f}: {1} loaded on {2}".format(self.simulator.simulator_time, entity.name,
                                                                 vehicle.name))
        entity = vehicle

        self.simulator.schedule_event_now(self, "exit_output_node", entity=entity)

    def exit_output_node(self, entity, **kwargs):
        # Add interarrival delay
        interarrival_delay = entity.interarrival_distribution(*entity.interarrival_times)
        self.simulator.schedule_event_rel(interarrival_delay, self, "exit_with_vehicle", entity=entity)

    def exit_with_vehicle(self, entity, **kwargs):
        logger.debug(
            "Time {0:.2f}: {1} exits {2}".format(self.simulator.simulator_time, entity.name, self.name))
        entity.next_link_port.enter_input_node(entity)

    def choose_next_link(self, entity):
        try:
            # Selection based on weights in links - to see which vehicle we need
            next_list = self.next if isinstance(self.next, list) else [self.next]
            # no circular behaviour - container cannot go back to previous port
            if len(next_list) > 1:
                next_list = [link for link in next_list if
                             link.destination not in entity.entities_on_vehicle[0].shipping_route]
                if len(next_list) == 0:
                    next_list = next_list
            elif len(next_list) == 1:
                next_list = next_list
            else:
                raise ValueError("{0} has no next process assigned".format(self.name))

            weights = [link.selection_weight for link in next_list]
            link_by_weight = np.random.choice(np.array(next_list), p=weights / np.sum(weights))
            return link_by_weight
        except ValueError:
            ValueError("{0} has no next process assigned".format(self.name))
        except AttributeError:
            try:
                if len(next_list) > 1:
                    next_process = np.random.choice(np.array(next_list))
                    return next_process
                elif len(next_list) == 1:
                    return self.next
            except AttributeError:
                raise AttributeError("{0} has no next process assigned".format(self.name))


class ImportPort(Port):
    def __init__(self, simulator, processing_time, capacity=1, distribution=np.random.triangular, **kwargs):
        super().__init__(simulator, capacity=capacity, distribution=distribution, processing_time=processing_time,
                         **kwargs)

        self.number_of_criminal_containers_checks = 0
        self.total_number_of_custom_checks = 0
        self.detection_probability = 0
        self.total_number_of_containers = 0
        self.total_parcels = 0
        self.seized_parcels = 0

    def determine_custom_check(self, entity: Container, **kwargs):
        # This is now random but can be changed to a more realistic distribution
        # assumption: around 10% of the risk containers are checked by customs
        # information flow of container
        entity.custom_checks_import_port = np.random.choice([True, False], p=[0.1, 0.9])

    def seize_resource(self, entity: [Product, Container], **kwargs):
        #already have to set the next link upfront to see whether it is the "end" import port or still a transit
        #and use the right vehicle
        try:
            # Selection based on weights in links
            next_list = self.next if isinstance(self.next, list) else [self.next]
            next_list = [link for link in next_list if
                         link.destination not in entity.entities_on_vehicle[0].shipping_route]
            # no circular behaviour - container cannot go back to previous port
            if len(next_list) > 1:
                next_list = [link for link in next_list if
                             link.destination not in entity.entities_on_vehicle[0].shipping_route]
            elif len(next_list) == 1:
                next_list = next_list
            else:
                raise ValueError("{0} has no next process assigned".format(self.name))

            weights = [link.selection_weight for link in next_list]
            entity.next_link_ports_imp = np.random.choice(np.array(next_list), p=weights / np.sum(weights))
        except AttributeError:
            try:
                if len(next_list) > 1:
                    entity.next_link_ports_imp = np.random.choice(np.array(next_list))
                elif len(next_list) == 1:
                    entity.next_link_ports_imp = next_list[0]
            except AttributeError:
                raise AttributeError("{0} has no next process assigned".format(self.name))

        if isinstance(entity.next_link_ports_imp.destination, Port):
            super().seize_resource(entity)
        else:
            if isinstance(entity, Container):
                entity.shipping_route.append(self)
                self.quantity += np.sum([products.quantity for products in entity.criminal_products_in_container])
                for product in entity.criminal_products_in_container:
                    self.arrivals_amount[product.name] = product.quantity
                    product.start_import_port = self.simulator.simulator_time
                    product.route.extend(entity.shipping_route)
                    product.shipping_route = [e for e in entity.shipping_route]
                    product.on_container_international_transport = entity
                    product.international_transport_time = self.simulator.simulator_time - \
                                                           product.start_international_transport
                    product.next_link_ports_imp = entity.next_link_ports_imp
                    product.travel_cost.update(entity.travel_cost)

                entity.start_import_port = self.simulator.simulator_time
            self.simulator.schedule_event_now(self, "load_off_containers_and_place_on_terminal", entity=entity)

    def load_off_containers_and_place_on_terminal(self, entity: Container, **kwargs):
        # time to unload and place one container at the steck (10 minutes per container) + time when it unloads from the boat
        time_per_container = np.random.exponential(10/60/24) if "import_time_load_off_per_container" not in self.kwargs else \
            np.random.exponential(self.kwargs["import_time_load_off_per_container"])
        time_when_unload = np.random.uniform(1/24, 30/24) if "import_time_load_off_wait" not in self.kwargs else \
            np.random.uniform(*self.kwargs["import_time_load_off_wait"])

        time_load_off = time_per_container*entity.bill_of_loading + time_when_unload

        self.simulator.schedule_event_rel(time_load_off, self, "wait_on_stack", entity=entity)

    def wait_on_stack(self, entity, **kwargs):
        self.total_number_of_containers += 1
        self.total_parcels += np.sum([len(p.parcels) for p in entity.criminal_products_in_container])
        if isinstance(entity, Product):
            print('what is going on')
        if entity.custom_checks_import_port == True:
            #custom checks
            # check often quite fast but delay of 12-14 hours for results of the scan
            time_waiting_stack_for_check = np.random.uniform(0.5, 1)
            extracting = np.random.choice([True, False], p=[0.5, 0.5]) if "import_prob_extracting" not in self.kwargs else \
                np.random.choice([True, False], p=[self.kwargs["import_prob_extracting"], 1-self.kwargs["import_prob_extracting"]])
            if extracting == True:
                extracted_product = entity.criminal_products_in_container
                entity.criminal = False
                for product in extracted_product:
                    product.custom_checks_import_port = entity.custom_checks_import_port
                    product.extracting_import_port = True
                    self.simulator.schedule_event_rel(time_waiting_stack_for_check, self, "enter_output_node",
                                                      entity=product)
                #Container without illegal products to check
                self.simulator.schedule_event_rel(time_waiting_stack_for_check, self, "check_by_custom", entity=entity)
            else:
                self.simulator.schedule_event_rel(time_waiting_stack_for_check, self, "check_by_custom", entity=entity)
        else:
            #transport of container
            time_waiting_stack_for_transport = np.random.uniform(0.5, 3) if "import_wait_on_steck_time" not in self.kwargs else \
                np.random.uniform(*self.kwargs["import_wait_on_steck_time"])
            self.simulator.schedule_event_rel(time_waiting_stack_for_transport, self, "enter_output_node",
                                              entity=entity)

    def check_by_custom(self, entity, **kwargs):
        self.total_number_of_custom_checks += 1
        if entity.criminal == True:
            self.number_of_criminal_containers_checks += 1
            self.seized_parcels += np.sum([len(p.parcels) for p in entity.criminal_products_in_container])
            logger.debug("Gotcha! -- {0} at simulation time {1:.1f}".format(entity.name,
                                                                           self.simulator.simulator_time))
            del entity
            # do not want to lose this data on travel times etc. -- so how to cope with it?
            # this is interesting for getting "online" information

        elif entity.criminal == False:
            del entity

        # self.detection_probability = \
        #     (self.number_of_criminal_containers_checks / self.total_number_of_custom_checks)

    def enter_output_node(self, entity: [Product, Container], **kwargs):
        if isinstance(entity, Container):
            self.quantity -= np.sum([products.quantity for products in entity.criminal_products_in_container])
            products_left = [products for products in entity.criminal_products_in_container]
            self.products_left.extend(products_left)
        else:
            self.quantity -= entity.quantity
            self.products_left.append(entity.name)

        #set vessel instead of truck when next is also a port
        if isinstance(entity.next_link_ports_imp.destination, Port):
            vehicle = Vessel(self.simulator)
        elif isinstance(entity.next_link_ports_imp, IntermodalTimeLink):
            # if intermodal or rail
            if "RAIL" in entity.next_link_ports_imp.modality:
                vehicle = Train(self.simulator)
            # if intermodal or truck
            elif "INTERMODAL" in entity.next_link_ports_imp.modality or "TRUCK" in entity.next_link_ports_imp.modality:
                vehicle = LargeTruck(self.simulator)
        else:
            # If there is a vehicle
            if "vehicle_type" in self.kwargs:
                if "vehicle_speed" in self.kwargs:
                    vehicle = self.kwargs["vehicle_type"](self.simulator, vehicle_speed=self.kwargs["vehicle_speed"])
                else:
                    vehicle = self.kwargs["vehicle_type"](self.simulator)

        vehicle.entities_on_vehicle.append(entity)
        vehicle.next_link_ports_imp = entity.next_link_ports_imp
        logger.debug("Time {0:.2f}: {1} loaded on {2} with destination {3}".format(self.simulator.simulator_time, entity.name,
                                                                 vehicle.name, entity.next_link_ports_imp.destination.name))
        entity = vehicle

        self.simulator.schedule_event_now(self, "exit_output_node", entity=entity)
    def exit_with_vehicle(self, entity, **kwargs):
        # logger.info(
        #     "Time {0:.2f}: {1} exits {2}".format(self.simulator.simulator_time, entity.name, self.name))
        entity.next_link_ports_imp.enter_input_node(entity)





class ResourceTransit(Resource):
    def __init__(self, id, simulator, queue, distribution, processing_time, transfer_in_time, **kwargs):
        super().__init__(id, simulator, queue, distribution, processing_time, transfer_in_time, **kwargs)

    def processing(self, **kwargs):
        if isinstance(self.distribution, type(np.random.RandomState().choice)):
            processing_time_dist = np.random.choice(self.processing_time[0], p=self.processing_time[1]) / 60 / 24
        else:
            try:
                dist_model = self.distribution(*self.processing_time)
                # When distribution fitted
                processing_time_dist = dist_model.rvs(random_state=self.simulator.seed) / 60 / 24
            except TypeError:
                try:
                    processing_time_dist = self.distribution.rvs(random_state=self.simulator.seed) / 60 / 24
                except TypeError:
                    processing_time_dist = self.distribution.rvs(*self.processing_time, random_state=self.simulator.seed) / 60 / 24

        if "processing_entity" in kwargs:
            self.processing_entity = kwargs["processing_entity"]

        self.simulator.schedule_event_rel(processing_time_dist, self, "exit_resource", **kwargs)





