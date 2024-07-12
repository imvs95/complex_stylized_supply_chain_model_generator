"""
Created on: 10-2-2023 16:22

@author: IvS
"""

import pandas as pd


def create_input_data_graph_from_excel(excel_file):
    """Create a dictionary from the input data of an Excel to create a supply chain graph.
    We use this for the establishment of the ground truth graph, and other dictionaries for graphs."""
    df_actors = pd.read_excel(excel_file, sheet_name="actors")

    actors = {}
    for name in df_actors["actor"]:
        df_actor = df_actors[df_actors["actor"] == name]
        params = df_actor.iloc[:, 1:].to_dict(orient="records")[0]
        params = {k: v for k, v in params.items() if v == v}
        if params["num"] == 0:
            continue
        elif params["num"] == 1:
            params = {k: [eval(v)] if isinstance(v, str) else v for k, v in
                      params.items()}
            # modify
            if "capacity" in params:
                params["capacity"] = [params["capacity"]]
            if "divide_quantity" in params:
                params["divide_quantity"] = [params["divide_quantity"]]
            actors[name] = params
        elif params["num"] > 1:
            params = {k: eval(v) if k != "num" else v for k, v in params.items()}
            actors[name] = params

    try:
        df_distances = pd.read_excel(excel_file, sheet_name="distances")
        # TODO add link type (?)
        distances = dict(
            zip(df_distances["links"].apply(lambda x: eval(x)), df_distances["distance_km"].apply(lambda x: eval(x))))
    except ValueError:
        distances = {}

    return actors, distances


def configure_input_data(a, d, num_supplier, num_manufacturer, num_export_port,
                         num_transit_port, num_import_port, num_wholesales_distributor,
                         num_retailer):
    """Configure the input data from the optimization model as two dictionaries
    used to create a graph. """
    #TODO make a,d self?


    return 1


if __name__ == "__main__":
    actors, distances = create_input_data_graph_from_excel(r"../input/actors_config.xlsx")

    configure_input_data(actors, distances, 3, 4, 1, 2, 3, 1, 5)




