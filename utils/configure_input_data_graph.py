"""
Created on: 19-6-2023 15:24

@author: IvS
"""

import pandas as pd
from copy import deepcopy

from structure_model_composer.sample import read_skeleton_data_base

def add_input_params_to_db_structures(db_structures, df_input_params):
    reshape_input_params = df_input_params.set_index("actor").to_dict(orient="index")
    dict_input_params_actors = {n: {k: v for k, v in d.items() if not pd.isna(v)} for n, d in reshape_input_params.items()}
    dict_input_params_actors = {n: {k: eval(v) if type(v) == str else v for k, v in d.items()}
                                for n, d in dict_input_params_actors.items()}

    new_db_structures = deepcopy(db_structures)
    for _, row in new_db_structures.iterrows():
        graph = row["graph"]

        # set location with location if not there
        actors = set(attr_dict.get("entity_type") for _, attr_dict in graph.nodes(data=True))
        counts_actors = {value: sum(1 for _, attr_dict in graph.nodes(data=True) if attr_dict.get("entity_type") == value) for
                  value in actors}
        alphabet = [chr(65 + i) for i in range(26)]  # List of uppercase letters

        alphabet_mapping = {}
        for key, value in counts_actors.items():
            if value > 0:
                alphabet_mapping[key] = alphabet[:value]

        for node, attr in graph.nodes(data=True):
            if "capacity" in attr:
                attr["capacity"] = int(attr["capacity"])

            if "id" in attr:
                attr["node_nr"] = attr["id"]
                del attr["id"]

            if "location" not in attr:
                graph.nodes[node]["location"] = alphabet_mapping[attr["entity_type"]][0]
                del alphabet_mapping[attr["entity_type"]][0]

            try:
                additional_attr = dict_input_params_actors[attr["entity_type"]]
                graph.nodes[node].update(additional_attr)
            except KeyError:
                continue

        row["graph"] = graph

    return new_db_structures


if __name__ == "__main__":
    db_structures = read_skeleton_data_base(r"../structure_model_composer/databases/test_location_graphs_100")
    default_input_params = pd.read_excel(r"../input/default_input_params_actors.xlsx", sheet_name="actors")

    new_db = add_input_params_to_db_structures(db_structures, default_input_params)

    print("Added input params to actors")