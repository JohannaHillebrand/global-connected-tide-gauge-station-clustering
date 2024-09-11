import json
import os

import geopandas
import networkx
import shapely
from loguru import logger

from src.inner.plot import plot_line_graph
from src.inner.sea_level_line_graph import check_for_consistency, group_stations, \
    sort_neighbors_for_nodes, gdf_from_graph
from src.inner.tide_gauge_station import TideGaugeStation


def start(all_stations, difference_file_name, difference_file_path, land_gdf,
          measuring_stations_gdf, output_path):
    # calculate a tree graph for each timespan and save it
    logger.info("Creating tree graph for each timespan")
    threshold = 0.030
    output_folder = os.path.join(output_path, f"longest_edge_removed/threshold_{threshold}/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    directories = [x[0] for x in os.walk(difference_file_path)][1:]
    for directory in directories:
        input_path = os.path.join(directory, difference_file_name)
        output_path = os.path.join(output_folder, directory.split("/")[-1])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        metadata_path = os.path.join(output_path, "metadata.txt")
        create_tree_graph(input_path, all_stations, measuring_stations_gdf, land_gdf, output_path,
                          metadata_path, threshold)


def create_tree_graph(in_path: str, stations: [TideGaugeStation], stations_gdf: geopandas.GeoDataFrame,
                      land_for_plotting: geopandas.GeoDataFrame, out_path: str,
                      metadata_path: str, threshold: float):
    """
    calculate a tree graph for the whole timespan
    :param threshold:
    :param metadata_path:
    :param out_path:
    :param in_path:
    :param stations:
    :param stations_gdf:
    :param land_for_plotting:
    :return:
    """
    # read in the sea level differences
    sea_level_diffs = read_sea_level_differences(in_path)
    # check for stations that are not in the sea level differences dictionary and remove them and give a warning
    # check for stations that are in the sea level differences dictionary but not in the list of stations,
    # and give a warning
    stations = check_for_consistency(stations, sea_level_diffs, metadata_path)
    # group stations, such that only a subset of nodes is there for comparison (save time)
    grouped_stations = group_stations(stations, metadata_path)
    # sort the neighbors for each node by the difference in sea level
    ordered_stations = sort_neighbors_for_nodes(sea_level_diffs, grouped_stations)
    # create graph containing nodes
    graph = networkx.Graph()
    for station in stations:
        graph.add_node(station.id, geometry=shapely.geometry.Point(station.longitude, station.latitude))
    # select the best neighbor for each node, unless a cycle is formed
    logger.info("Creating tree graph")
    change = True
    counter = 0
    how_many_edges_removed = 0
    for node in graph.nodes():
        if len(ordered_stations[node]) == 0:
            continue
        while True:
            if len(ordered_stations[node]) == 0:
                break
            second_node = ordered_stations[node].pop(0)
            # check if diff is too large
            diff = second_node[0]
            if diff > threshold:
                continue
            graph.add_edge(node, second_node[1])
    while not networkx.is_forest(graph):
        cycle_edges = networkx.find_cycle(graph)
        longest_edge = 0.0
        edge_to_remove = None
        for edge in cycle_edges:
            first_node = edge[0]
            second_node = edge[1]
            diff = sea_level_diffs[first_node][second_node]
            if diff > longest_edge:
                longest_edge = diff
                edge_to_remove = edge
        if edge_to_remove is not None:
            how_many_edges_removed += 1
            graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
    logger.info(f"Number of components in graph: {networkx.number_connected_components(graph)}")
    logger.info(f"Number of edges in the graph: {len(graph.edges)}")
    logger.info(f"Removed {how_many_edges_removed} edges")

    # while change:
    #     counter += 1
    #     graph, change = select_best_neighbors_tree_graph(graph, ordered_stations, threshold)
    #     with open(metadata_path, "a") as metadata_file:
    #         metadata_file.write(f"Iteration: {counter}\n")
    #         metadata_file.write(f"Number of components in graph: {networkx.number_connected_components(graph)}\n")

    with open(metadata_path, "a") as metadata_file:
        metadata_file.write(f"Graph after assigning edges: {graph}\n")
        metadata_file.write(f"Number of components in graph: {networkx.number_connected_components(graph)}\n")

    # check if there are cycles in the graph
    try:
        networkx.find_cycle(graph)
        logger.warning(
            f"There are still cycles in the graph. This should not be the case and points to an error in the code.")
    except networkx.exception.NetworkXNoCycle:
        pass
    logger.info(f"Finished creating tree graph")
    # save node-data
    node_data = {}
    for station in stations:
        node_data[station.id] = [station.longitude, station.latitude]
    json.dump(node_data, open(os.path.join(out_path, "node_data.json"), "w"))
    # save graph
    networkx.write_edgelist(graph, os.path.join(out_path, "tree_graph.csv"),
                            delimiter=";", data=False)
    # plot results
    result_gdf = gdf_from_graph(graph)
    present_stations_gdf = stations_gdf[stations_gdf["id"].isin(graph.nodes)]
    plot_line_graph(land_for_plotting, out_path, present_stations_gdf, result_gdf, "tree_graph")
    with open(os.path.join(out_path, "differences.json"), "w") as diffs_file:
        json.dump(sea_level_diffs, diffs_file)
    return


def read_sea_level_differences(path: os.path):
    """
    Read in the differences in sea level between measuring stations
    :param path:
    :return:
    """
    # read in difference data between measuring stations
    with open(path, "r") as json_file:
        differences = json.load(json_file)
    # replace strings in differences with ints (these are the keys in the dictionary)
    differences_int = {}
    for old_key in differences.keys():
        new_key = int(old_key)
        sub_dictionary = {}
        for old_subkey in differences[old_key].keys():
            new_subkey = int(old_subkey)
            sub_dictionary[new_subkey] = differences[old_key][old_subkey]
        differences_int[new_key] = sub_dictionary
    return differences_int
