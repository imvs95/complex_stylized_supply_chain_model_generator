import pandas as pd
import matplotlib.pyplot as plt
import collections
import networkx as nx


def draw_graph(results):
    """
    Function used to draw randomly generated graph when finished.
    """
    if isinstance(results, nx.DiGraph):
        G = results
    else:
        G = results[1]["graph"]

    pos_values = [node["pos"][0] for id, node in G.nodes(data=True)]
    counter = collections.Counter(pos_values)
    projected = {}
    for key, value in counter.items():
        projected[key] = 1
    for i in range(len(G.nodes())):
        G.nodes()[i]["color"] = (
            0 if G.nodes()[i]["echelon"] == 0 else 1 / G.nodes()[i]["echelon"]
        )
        k = projected[G.nodes()[i]["pos"][0]]
        y = counter[G.nodes()[i]["pos"][0]]
        G.nodes()[i]["pos"] = (G.nodes()[i]["pos"][0], k / y)
        projected[G.nodes()[i]["pos"][0]] += 1
    pos = nx.get_node_attributes(G, "pos")
    color_values = [node["color"] for id, node in G.nodes(data=True)]
    fig, ax = plt.subplots()
    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        cmap=plt.get_cmap("viridis"),
        node_color=color_values,
        with_labels=True,
        font_color="orange",
        font_size=9,
    )
    fig.tight_layout()
    return fig


def convert_results(results):
    """This function converts the results dictionary to a pandas dataframe for easier analysis and plotting."""

    for replication in results.keys():
        df = pd.DataFrame()
        for key, value in results[replication]["time_series"].items():
            data = pd.Series(value, name=key)
            df = pd.concat([df, data], axis=1)
        results[replication]["time_series"] = df

    return results


def plot_timeseries_per_type(results):
    """ This function plots the time series of each entity type in the supply chain."""
    results = convert_results(results)
    # Structure results
    if len(results.keys()) > 1:
        df = pd.DataFrame()
        for key, value in results.items():
            data = value["time_series"]
            data["replication"] = [key for i in range(len(data))]
            df = pd.concat([df, data], axis=0)
    else:
        df = results[1]["time_series"]

    # Establish dict with lists for each entity type
    columns = list(df.columns)
    columns.remove("replication")

    data1 = {
        "supplier": [],
        "manufacturer": [],
        "export_port": [],
        "transit_port": [],
        "import_port": [],
        "wholesaler": [],
        "retailer": [],
    }

    types = []
    for type_ in data1.keys():
        types.append(type_)

    for entity in columns:
        for type_ in types:
            if entity.split("_", 1)[1] == type_:
                data1[type_].append(entity)

    print(data1)
    if len(results.keys()) == 1:
        fig, ax = plt.subplots(len(types), figsize=(8, 15))
        n = 0
        for key, value in data.items():
            df[value].plot(ax=ax[n], legend=False)
            ax[n].set_title(key)
            n += 1

    if len(results.keys()) > 1:
        df = df.reset_index()
        df = df.rename(columns={"index": "step"})
        fig, ax = plt.subplots(len(types), figsize=(8, 15))

        n = 0
        print(df)
        grouped_mean = df.groupby(["step", "replication"]).mean()
        for key, value in data1.items():
            grouped_mean[value].plot(ax=ax[n], legend=False)
            ax[n].set_title(key)
            n += 1
            print(n)

    fig.tight_layout()
    return fig


def calc_average(results):
    """ This function calculates the average of the time series of each entity type in the supply chain."""
    results = convert_results(results)
    # Structure results
    if len(results.keys()) > 1:
        df = pd.DataFrame()
        for key, value in results.items():
            data = value["time_series"]
            data["replication"] = [key for i in range(len(data))]
            df = pd.concat([df, data], axis=0)
    else:
        df = results[1]["time_series"]

    # Establish dict with lists for each entity type
    columns = list(df.columns)
    columns.remove("replication")

    data1 = {
        "supplier": [],
        "manufacturer": [],
        "export_port": [],
        "transit_port": [],
        "import_port": [],
        "wholesaler": [],
        "retailer": [],
    }

    types = []
    for type_ in data1.keys():
        types.append(type_)

    for entity in columns:
        for type_ in types:
            if entity.split("_", 1)[1] == type_:
                data1[type_].append(entity)

    if len(results.keys()) > 1:
        df = df.reset_index()
        df = df.rename(columns={"index": "step"})
        grouped_mean = df.groupby(["step"]).mean()

    return grouped_mean


def calc_outcomes_sum(results):
    """ This function calculates the sum of the time series of each entity type in the supply chain."""
    results = convert_results(results)
    # Structure results
    if len(results.keys()) > 1:
        df = pd.DataFrame()
        for key, value in results.items():
            data = value["time_series"]
            data["replication"] = [key for i in range(len(data))]
            df = pd.concat([df, data], axis=0)
    else:
        df = results[1]["time_series"]

    # Establish dict with lists for each entity type
    columns = list(df.columns)
    columns.remove("replication")

    data1 = {
        "supplier": [],
        "manufacturer": [],
        "export_port": [],
        "transit_port": [],
        "import_port": [],
        "wholesaler": [],
        "retailer": [],
    }

    types = []
    for type_ in data1.keys():
        types.append(type_)

    for entity in columns:
        for type_ in types:
            if entity.split("_", 1)[1] == type_:
                data1[type_].append(entity)

    if len(results.keys()) > 1:
        df = df.reset_index()
        df = df.rename(columns={"index": "step"})
        grouped_mean = df.groupby(["step"]).mean()

    outcomes = pd.DataFrame()

    for key in data1.keys():
        outcomes[key] = grouped_mean[data1[key]].sum(axis=1)

    return outcomes
