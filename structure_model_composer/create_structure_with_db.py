"""
Created on: 3-5-2023 17:38

@author: IvS, Bruno Hermans
"""


import json
import time
import logging
import matplotlib.pyplot as plt
from structure_model_composer.structure_generator import StructureGenerator
from structure_model_composer.sample import (
    create_graph_hashes,
    save_skeletons_batch,
    read_skeleton_data_base,
)
from structure_model_composer.process_results import draw_graph


def fill_db(n=1, n_structures=1, db_name="db", constraints_file="../input/constraints_sc_nodoubleinput_cnhk_usa.json"):
    """This function fills the database with graph structures given the constraints file.

    Parameters:
        n (int): number of structures to generate
        n_structures (int): number of structures to generate per batch
        db_name (str): name of the database
        constraints_file (str): path to the constraints file

    Returns:
        db: database with structures, stored on the disk
    """

    if db_name == "db":
        db_name = "db_"+str(n)
    with open(constraints_file) as f:
        res = json.load(f)
    composer = StructureGenerator(res)
    start_time = time.time()
    structures_generated = 0
    isomorphic_rates = []
    for _ in range(n):
        df = composer.create_graphset_df(n=n_structures, coordinates=False)
        df = create_graph_hashes(df)
        structures_generated += n_structures
        print(
            "----------------------------------------------------------------------------"
        )
        print(
            "10 structures generated. Saving structures --- %s seconds ---"
            % (time.time() - start_time)
        )
        print("Total of {} structures generated".format(structures_generated))
        save_skeletons_batch(df, dbname=db_name)
        print("Results saved to DB. --- %s seconds --- " % (time.time() - start_time))
        df1 = read_skeleton_data_base(dbname=db_name)
        isomorphic_rate = len(df1.index) / structures_generated
        isomorphic_rates.append(
            {
                "structures_generated": structures_generated,
                "isomorphic_rate": isomorphic_rate,
            }
        )
        print(
            "df1 {}, structures generated {}".format(
                len(df1.index), structures_generated
            )
        )
        print("Isomorphic rate is {}".format(isomorphic_rate))
        print(
            "----------------------------------------------------------------------------"
        )
        # pd.DataFrame(isomorphic_rates).to_csv("Isomorphics.csv")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fill_db(n=10, db_name="test_10")
    df = read_skeleton_data_base("test_10")

    # draw_graph(df["graph"].iloc[-1])
    # plt.show()
    print(df)

