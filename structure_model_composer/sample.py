import shelve
import random
# from sklearn.cluster import KMeans
import networkx as nx
import pandas as pd


def create_graph_hashes(df):
    """Function to create a hash of a model structure.

    Args:
        df (pd.DataFrame): DataFrame of model structures that need to be hashed.

    Returns:
        df (pd.DataFrame): Dataframe with hashes.
    """
    df["hash"] = df["graph"].apply(
        lambda x: nx.weisfeiler_lehman_graph_hash(x, edge_attr=None, node_attr=None)
    )
    return df


def save_skeletons_batch(df, dbname="shelfE"):
    """Function that saves model structures in a database

    Args:
        df (pd.DataFrame): df with model structures and properties
    """
    # store objects in list
    objects = []
    for index, row in df.iterrows():
        objects.append(
            {
                "hash": row["hash"],
                "graph": row["graph"],
                "edges": row["edges"],
                "nodes": row["nodes"],
                "betweenness": row["betweenness"],
                "degree_centrality": row["degree_centrality"],
                "closeness_centrality": row["closeness_centrality"],
            }
        )
    # store in shelve

    with shelve.open(dbname, "c") as shelf:
        for object in objects:
            id = random.randint(1, 10000000)
            key = object["hash"] + "_" + str(row["betweenness"]) + "_" + str(id)
            shelf[key] = object


def read_skeleton_data_base(dbname="shelfE"):
    """reads shelf database into dataframe

    Args:
        dbname (str, optional): Name of database to read. Defaults to "shelfE".

    Returns:
        df: df with model structures
    """
    objects = []
    with shelve.open(dbname, "r") as shelf:
        for key in shelf.keys():
            objects.append(shelf[key])

    return pd.DataFrame(objects)


def save_results(results_dict, dbname="results", run_name="default_name"):
    """Function to save results from a simulation run.

    Args:
        results_dict (_type_): Dict with simulation results.
        dbname (str, optional): Name of database to save results in. Defaults to "results".
        run_name (str, optional): Name of run. Defaults to "default_name".
    """
    with shelve.open(dbname, "c") as shelf:
        for key, value in results_dict.items():
            shelf[str(key) + "_" + run_name] = value


def read_results(dbname="results"):
    """Function used to read simulation results

    Args:
        dbname (str, optional): Name of results database. Defaults to "results".

    Returns:
        dict: Dictionary with results.
    """
    results_dict = {}
    with shelve.open(dbname, "r") as shelf:
        for key in shelf.keys():
            object = shelf[key]
            results_dict[key] = object
    return results_dict


# def kmeans_sampling(df, n=1):
#     """Function that samples models structures from a pandas dataframe.
#
#     Args:
#         df (pd.DataFrame): Dataframe with model structures to be sampled from.
#         n (int, optional): Number of structures to sample from dataframe. Defaults to 1.
#
#     Returns:
#         _type_: _description_
#     """
#     X = df.copy()
#     columns = [
#         "edges",
#         "nodes",
#         "betweenness",
#         "degree_centrality",
#         "closeness_centrality",
#     ]
#     X = X[columns]
#     kmeans = KMeans(n_clusters=n).fit(X)
#     X.loc[:, "cluster"] = kmeans.labels_
#     X.sort_values("cluster")
#     sample = df.loc[X.groupby("cluster").sample(n=1).index]
#     return sample


def merge_db(n_jobs=10, name="db", path="/structures", saveas="structure_db"):
    """Function used to merge databases.

    Args:
        n_jobs (int, optional): Number of databases to merge. Defaults to 10.
        name (str, optional): Base of databases names to merge. Defaults to "db".
        path (str, optional): Folder location with databases to merge. Defaults to "/structures".
        saveas (str, optional): Name of new database. Defaults to "structure_db".
    """
    merged = pd.DataFrame()
    for i in range(n_jobs):
        db_name = name + str(i)
        db_path = path + "/" + db_name
        print(db_path)
        df = read_skeleton_data_base(db_path)
        merged = pd.concat([merged, df], axis=0)
    save_skeletons_batch(merged, saveas)


def calc_isomorphic_rate(hash_table, d_rate=1):
    """Function that computes the isomorphic rate.

    Args:
        hash_table (str): location of file with hashes
        d_rate (int, optional): Times to compute the iso rate. Defaults to 1.

    Returns:
        list: list with isomorphic rates.
    """
    df = pd.read_csv(hash_table, header=None, names=["hash"])
    hashes = df["hash"].to_list()
    steps = len(hashes) / d_rate
    iso_rates = []
    for i in range(round(steps)):
        end = d_rate * i
        if end <= len(hashes) and end != 0:
            hash_selection = hashes[0:end]
            iso_rate = len(set(hash_selection)) / len(hash_selection)
            iso_rates.append(iso_rate)
    return iso_rates
