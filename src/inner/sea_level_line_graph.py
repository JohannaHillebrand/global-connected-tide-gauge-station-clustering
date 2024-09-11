# IDEA: iterate through each linegraph and check for each edge, if it is the best one (in regard to the difference
# in sea level)
import json
import math
import os

import geopandas
import networkx
import searoute as sr
import shapely
from loguru import logger

from src.inner import tide_gauge_station, timeseries_difference
from src.inner.plot import plot_line_graph, random_color_generator
from src.inner.tide_gauge_station import TideGaugeStation


def gdf_from_graph(graph: networkx.Graph):
    """
    Creates a gdf from a graph, where all edges form a line that is assigned a color
    :param graph:
    :return:
    """
    edge_dict = {}
    edge_counter = 0
    color_counter = 0
    color_list = random_color_generator(500)
    # color_list = ["darkorchid", "darkblue", "teal", "chartreuse", "yellow", "lightseagreen", "firebrick",
    # "yellowgreen",
    #               "fuchsia", "dodgerblue", "gold", "seagreen", "mediumslateblue", "deeppink", "lightcoral", "indigo",
    #               "slategrey", "rosybrown", "peru", "palevioletred", "purple", "darkturquoise", "mediumaquamarine",
    #               "tomato", "navy", "thistle", "plum", "darkorange", "darkolivegreen", "darkslateblue", "darkviolet",
    #               "darkgreen", "blueviolet", "darkmagenta", "darkred", "darkcyan", "darkkhaki", "darkgoldenrod",
    #               "darkslategray", "darkseagreen", "blue", "green", "red", "cyan", "magenta", "yellow", "black",
    #               "pink", "brown", "orange", "gray", "purple", "olive", "maroon", "green", "lime", "aqua", "silver",
    #               "teal", "navy", "fuchsia"]
    components = networkx.connected_components(graph)
    edge_id = []
    geometry = []
    colors = []
    for component in components:
        for edge in networkx.subgraph(graph, component).edges:
            node1 = edge[0]
            node2 = edge[1]
            geometry1 = graph.nodes[node1]["geometry"]
            geometry2 = graph.nodes[node2]["geometry"]
            lon1 = geometry1.x
            lon2 = geometry2.x
            if abs(lon1 - lon2) > 180:
                continue
            edge_line = shapely.geometry.LineString(
                [graph.nodes[node1]["geometry"], graph.nodes[node2]["geometry"]])
            edge_id.append(edge_counter)
            geometry.append(edge_line)
            colors.append(color_list[color_counter])
            edge_counter += 1
        color_counter += 1
    edge_dict = {"id": edge_id, "geometry": geometry, "color": colors}
    gdf_containing_edges = geopandas.GeoDataFrame.from_dict(edge_dict, orient="columns")
    return gdf_containing_edges


def group_stations(stations: [TideGaugeStation], metadata_path: str):
    """
    Sort the tide gauge station roughly, such that for each station we have a list of the closest stations to it
    One degree of latitude is approximately 111 kilometers (69 miles) at any latitude.
    One degree of longitude spans 111 kilometers (69 miles) at the equator and zero at the poles.
    The distance between degrees of longitude shrinks by a factor equal to the cosine of the latitude.
    Find all stations that are within 20 degrees of latitude (so 10 to each side) and at the equator also 10 degrees
    of longitude. This is a square of 2220 km x 2220 km. This is a rough estimate, as the earth is not a perfect
    sphere.
    To adjust for the shrinking of the longitude, we multiply the longitude difference with the cosine of the
    latitude.
    :param stations:
    :param metadata_path:
    :return:
    """
    grouping_stations = {}
    for station in stations:
        grouping_stations[station.id] = []
        for second_station in stations:
            if station.id != second_station.id:

                if (abs(station.latitude - second_station.latitude) < 10) and (abs(abs(
                        station.longitude - second_station.longitude) * math.cos(
                    math.radians(abs(station.latitude)))) < 10):
                    grouping_stations[station.id].append(second_station)
    if len(grouping_stations.keys()) > 0:
        with open(metadata_path, "a") as metadata_file:
            metadata_file.write(
                f"Preliminary grouping done. Averaged group size is: "
                f"{sum([len(grouping_stations[station]) for station in grouping_stations.keys()]) / len(grouping_stations.keys())} \n")
    return grouping_stations


def find_closest_end_node(data: []):
    """
    Find the closest node to the current node that has a degree of 1
    :param data:
    :return:
    """
    node1 = data[0]
    graph = data[1].copy()
    edge_nodes = data[2]

    closest_node = None
    closest_distance = float("inf")
    for node2 in edge_nodes:
        if graph.degree(node2) > 2:
            continue
        if node1 == node2:
            continue
        if graph.has_edge(node1, node2):
            continue
        graph.add_edge(node1, node2)
        if networkx.is_forest(graph):
            # check distance between two nodes
            # searoute uses long lat
            first_station_geometry = [graph.nodes[node1]["geometry"].x, graph.nodes[node1]["geometry"].y]
            second_station_geometry = [graph.nodes[node2]["geometry"].x, graph.nodes[node2]["geometry"].y]
            route = sr.searoute(first_station_geometry, second_station_geometry)
            dist = route.properties["length"]
            if dist < 1000 and dist < closest_distance:
                closest_distance = dist
                closest_node = node2
        graph.remove_edge(node1, node2)
    if closest_node is not None:
        return [node1, closest_node]
    else:
        return []


def sort_edges_to_add(edges_to_add: [], graph: networkx.Graph, sea_level_diffs: {int: {int: float}}):
    """
    Sort the edges that could be added to the graph, by removing those that are geographically very far away,
    and then sorting them according to their sea level behavior similarity
    :param sea_level_diffs:
    :param edges_to_add:
    :param graph:
    :return:
    """
    sorted_edges = []
    edges_to_sort = []

    for edge in edges_to_add:
        if edge == []:
            continue
        # check distance between two nodes
        route = sr.searoute((graph.nodes[edge[0]]["geometry"].x, graph.nodes[edge[0]]["geometry"].y),
                            (graph.nodes[edge[1]]["geometry"].x, graph.nodes[edge[1]]["geometry"].y))
        distance = route.properties["length"]
        if distance <= 1000:
            sea_level_difference = sea_level_diffs[edge[0]][edge[1]]
            edges_to_sort.append([edge, sea_level_difference])
    edges_to_sort.sort(key=lambda x: x[1])

    for edge in edges_to_sort:
        sorted_edges.append(edge[0])
    return sorted_edges


def merge_line_graphs_that_are_close(graph: networkx, sea_level_diffs: {int: {int: float}}):
    """
    Merge line graphs that are close to each other
    :param sea_level_diffs:
    :param graph:
    :return:
    """
    edge_nodes = []
    triples = []
    for node in graph.nodes():
        if graph.degree(node) <= 1:
            edge_nodes.append(node)

    # checke if two line graph ends are close to each other, so they can be joined. Beware of cycles
    edges_to_add = []
    # This does not work as intended, as we might get edges that are connected anyway, and we have only one option
    # per node that only has 1 neighbor. It would be better to sort all possible nodes, and then check, if adding the
    # best one would create a cycle, if yes, take the next best one
    # for node in edge_nodes:
    #     triples.append([node, graph, edge_nodes])
    # with Pool() as p:
    #     edges_to_add = p.map(find_closest_end_node, triples)

    # add all possible edges between end nodes
    for node in edge_nodes:
        for second_node in edge_nodes:
            if node == second_node:  # check if the two nodes are the same
                continue
            if graph.has_edge(node, second_node):  # check if the edge is already in the graph
                continue
            if [second_node, node] in edges_to_add:  # check if the edge is already in the list
                continue
            if second_node in networkx.algorithms.descendants(graph,
                                                              node):  # check if the two nodes are already connected
                continue
            edges_to_add.append([node, second_node])

    edges_to_add = sort_edges_to_add(edges_to_add, graph, sea_level_diffs)

    # add edges, if possible (no node is allowed to have more than two neighbors)
    for edge in edges_to_add:
        if edge == []:
            continue
        if graph.degree(edge[0]) <= 1 and graph.degree(edge[1]) <= 1:
            graph.add_edge(edge[0], edge[1])
        if graph.degree(edge[0]) > 2 or graph.degree(edge[1]) > 2:
            graph.remove_edge(edge[0], edge[1])
        if networkx.is_forest(graph):
            continue
        else:
            graph.remove_edge(edge[0], edge[1])
    return graph


def create_line_graph(stations_with_groups: {int: []}, differences: {int: {int: float}},
                      stations: [TideGaugeStation],
                      metadata_path: str):
    """
    Create a line graph from the stations. For each node check nodes that are in a certain distance form a line graph
    by searching the two nodes that are closest to each other and have the least difference in sea level.
    Calculate the two best edges for each node
    Add all edges to the graph, if there is a node with a degree > 2, check which edge is the worst and remove it
    New idea: for each node, order the other nodes that are close enough by the difference in sea level. Then for
    each node try to add the best edge to the graph, if the destination-node already has a degree of 2, take the next
    best edge and continue in this fashion
    :param metadata_path:
    :param stations:
    :param stations_with_groups:
    :param differences:
    :return:
    """
    graph = networkx.Graph()
    # add all nodes to the graph
    for station in stations:
        graph.add_node(station.id, geometry=shapely.geometry.Point(station.longitude, station.latitude))
    with open(metadata_path, "a") as metadata_file:
        metadata_file.write(f"Number of nodes in graph: {len(graph.nodes)}\n")

    # new idea - faster
    # TODO check via searoute is the geographical distance is small enough (some kind of threshold) ?
    # sort the neighbors for each node by the difference in sea level
    ordered_stations = sort_neighbors_for_nodes(differences, stations_with_groups)

    # Each node can choose one neighbor to connect to, if this neighbor has a degree < 2, otherwise select the next
    # best one and so on
    graph_assignment, change = select_best_neighbors(graph, ordered_stations)
    # Now, there might be nodes that only have one neighbor, so allow for choosing a neighbor for the second time (
    # once again only choose neighbors that have a degree < 2)
    counter = 0
    change = True
    while change:
        counter += 1
        graph_assignment, change = select_best_neighbors(graph_assignment, ordered_stations)
    with open(metadata_path, "a") as metadata_file:
        metadata_file.write(f"Number of iterations for assigning edges: {counter}\n")
    # check for cycles in the graph
    try:
        networkx.find_cycle(graph_assignment)
        logger.warning(
            f"There are still cycles in the graph. This should not be the case and points to an error in the code.")
    except networkx.exception.NetworkXNoCycle:
        pass

    merged_graph = merge_line_graphs_that_are_close(graph_assignment, differences)

    try:
        networkx.find_cycle(merged_graph)
        logger.warning(
            f"There are still cycles in the graph. This should not be the case and points to an error in the code.")
    except networkx.exception.NetworkXNoCycle:
        pass
    return merged_graph


def sort_neighbors_for_nodes(differences: {int: {int: float}}, stations_with_groups: {int: []}):
    """
    Sort the neighbors for each node by the difference in sea level
    :param differences:
    :param stations_with_groups:
    :return:
    """
    ordered_stations = {}
    for station in stations_with_groups.keys():
        ordered_stations[station] = []
        for second_station in stations_with_groups[station]:
            if second_station.id == station:
                continue
            if second_station.id in differences[station].keys():
                ordered_stations[station].append([differences[station][second_station.id], second_station.id])
        ordered_stations[station].sort(key=lambda x: x[0])
    return ordered_stations


def select_best_neighbors(graph: networkx.Graph, ordered_stations: {int: []}):
    """
    For each node, select the best neighbor to connect to, if this neighbor has a degree < 2
    :param graph:
    :param ordered_stations:
    :return:
    """
    counter = 0
    changes = False
    for node in graph.nodes():
        if graph.degree(node) >= 2:
            continue
        while True:
            if len(ordered_stations[node]) == 0:
                counter += 1
                break
            second_node = ordered_stations[node].pop(0)
            if graph.degree(second_node[1]) < 2 and not graph.has_edge(node, second_node[1]):
                # check distance via searoute, to avoid adding stations that are only close via a route that goes
                # over land
                first_station_geometry = [graph.nodes[node]["geometry"].x, graph.nodes[node]["geometry"].y]
                second_station_geometry = [graph.nodes[second_node[1]]["geometry"].x,
                                           graph.nodes[second_node[1]]["geometry"].y]
                route = sr.searoute(first_station_geometry, second_station_geometry)
                distance = route.properties["length"]
                if distance > 600:
                    continue
                else:
                    graph.add_edge(node, second_node[1])
                    try:  # check if adding this edge would create a cycle
                        networkx.find_cycle(graph)
                        cycle_exists = True
                    except networkx.exception.NetworkXNoCycle:
                        cycle_exists = False
                    if cycle_exists:
                        graph.remove_edge(node, second_node[1])
                        continue
                    changes = True
                    break
    return graph, changes


def check_for_consistency(stations: [TideGaugeStation], sea_level_diffs: {int: {int: float}}, metadata_path: str):
    """
    Check for stations that are not in the sea level differences dictionary and remove them and give a warning.
    Check for stations that are in the sea level differences dictionary but not in the list of stations, and give a
    warning.
    :param metadata_path:
    :param stations:
    :param sea_level_diffs:
    :return:
    """
    missing_stations = []
    for current_station in sea_level_diffs.keys():
        if current_station not in [this_station.id for this_station in stations]:
            missing_stations.append(current_station)
    if len(missing_stations) > 0:
        with open(metadata_path, "a") as metadata_file:
            metadata_file.write(
                f"Station(s) {[current_station for current_station in missing_stations]} is/are in the sea level "
                f"differences dictionary but not in the list of stations, this might lead to problems later on. "
                f"Proceeding...\n")

    stations_to_remove = []
    for this_station in stations:
        if this_station.id not in sea_level_diffs.keys():
            stations_to_remove.append(this_station)
    if len(stations_to_remove) > 0:
        with open(metadata_path, "a") as metadata_file:
            metadata_file.write(
                f"There are stations that are not in the sea level differences dictionary: Removing {[this_station.id
                                                                                                      for
                                                                                                      this_station in
                                                                                                      stations_to_remove]} and Proceeding...\n")
            metadata_file.write(f"Number of stations before removal: {len(stations)}\n")
        for current_station in stations_to_remove:
            stations.remove(current_station)
        with open(metadata_path, "a") as metadata_file:
            metadata_file.write(f"Number of stations after removal: {len(stations)}\n")
    return stations


def save_and_plot_line_graph(land_for_plotting: str, name: str, out_directory: str,
                             result_graph: networkx.Graph, stations: {int: TideGaugeStation}):
    """
    Save the line graph and plot it
    :param land_for_plotting:
    :param name:
    :param out_directory:
    :param result_graph:
    :param stations:
    :return:
    """
    stations_gdf = tide_gauge_station.gdf_from_stations(stations)
    node_data = {}
    for station in stations.values():
        node_data[station.id] = [station.longitude, station.latitude]
    with open(os.path.join(out_directory, "node_data.txt"), "w") as file:
        for node in node_data.keys():
            file.write(f"{node}: {node_data[node]}\n")
    result_gdf = gdf_from_graph(result_graph)
    networkx.write_edgelist(result_graph, os.path.join(out_directory, f"{name}.csv"),
                            delimiter=";", data=False)
    present_stations_gdf = stations_gdf[stations_gdf["id"].isin(result_graph.nodes)]
    plot_line_graph(geopandas.read_file(land_for_plotting), out_directory, present_stations_gdf, result_gdf, name)


def read_line_graph(edge_list_path: str, node_path: str):
    """
    Read in a line graph from a csv file
    :param node_path:
    :param edge_list_path:
    :return:
    """
    graph = networkx.Graph()
    # add nodes
    with open(node_path, "r") as node_file:
        for line in node_file:
            node = line.split(":")[0]
            coordinates = line.split(":")[1].replace("[", "").replace("]", "").replace("\n", "").split(",")
            graph.add_node(int(node), geometry=shapely.geometry.Point(float(coordinates[0]), float(coordinates[1])))
    # add edges
    with open(edge_list_path, "r") as edge_file:
        for line in edge_file:
            edge = line.split(";")
            graph.add_edge(int(edge[0]), int(edge[1]))
    return graph


def construct_line_graph_for_current_timestep(metadata_path: str, sea_level_differences: {},
                                              working_stations: [TideGaugeStation]):
    """
    Construct the line graph for the current timestep
    :param metadata_path:
    :param sea_level_differences:
    :param working_stations:
    :return:
    """
    working_stations = check_for_consistency(working_stations, sea_level_differences, metadata_path)
    grouped_stations = group_stations(working_stations, metadata_path)
    result_graph = create_line_graph(grouped_stations, sea_level_differences, working_stations, metadata_path)
    # write metadata to file
    with open(metadata_path, "a") as metadata_file:
        metadata_file.write(f"Number of components in graph: {networkx.number_connected_components(result_graph)}\n ")
    return result_graph


def reduce_line_graph(graph: networkx.Graph, current_stations: [TideGaugeStation],
                      metadata_path: str):
    """
    remove nodes that are not present in a given timespan
    :return:
    """
    graph = graph.copy()
    nodes_to_remove = []
    # find all nodes that do not exist in the current timespan
    for node in graph.nodes():
        if node not in [station.id for station in current_stations]:
            nodes_to_remove.append(node)

    # remove nodes and add edge
    for node in nodes_to_remove:
        if graph.degree(node) <= 1:
            graph.remove_node(node)
        else:
            neighbors = [x for x in graph.neighbors(node)]
            graph.add_edge(neighbors[0], neighbors[1])
            graph.remove_node(node)
    with open(metadata_path, "a") as metadata_file:
        metadata_file.write(f"Graph: {graph}\n")
        metadata_file.write(f"Number of components in graph: {networkx.number_connected_components(graph)}\n")

    return graph


def remove_stations_from_real_data(stations: [TideGaugeStation], simulated_differences: {int: {int: float}},
                                   real_differences: {int: {int: float}}):
    """
    Remove stations that are not present in the timespan between 2010 and 2020
    :param stations:
    :param simulated_differences:
    :param real_differences:
    :return:
    """
    # find all stations that are not present in the timespan between 2010 and 2020
    print(len(stations))
    stations_to_remove = []
    for station in stations:
        if station.id not in real_differences.keys() or station.id not in simulated_differences.keys():
            stations_to_remove.append(station)
    # remove stations
    for station in stations_to_remove:
        stations.remove(station)
        if station.id in simulated_differences.keys():
            simulated_differences.pop(station.id)
    for station in simulated_differences.keys():
        for remove_station in stations_to_remove:
            if remove_station.id in simulated_differences[station].keys():
                simulated_differences[station].pop(remove_station.id)

    print(len(stations))

    return stations, simulated_differences


def create_simulated_line_graph_based_on_real_data(all_stations: [TideGaugeStation],
                                                   sea_level_differences: {int: {int: float}},
                                                   real_sea_level_diffs: {int: {int: float}},
                                                   land_gdf: geopandas.GeoDataFrame,
                                                   measuring_stations_gdf: geopandas.GeoDataFrame, out_directory: str):
    """
    Create a line graph for the simulated data in which only the stations are used, that also exist in the real
    dataset
    :param out_directory:
    :param all_stations:
    :param sea_level_differences:
    :param real_sea_level_diffs:
    :param land_gdf:
    :param measuring_stations_gdf:
    :return:
    """
    working_stations, sea_level_diffs = remove_stations_from_real_data(all_stations, sea_level_differences,
                                                                       real_sea_level_diffs)
    # save diff-file
    with open(
            "../output/RMS/line_graph_2010_2020_simulated_data_remove_stations_not_present_in_real_data/differences"
            ".json",
            "w") as diffs_file:
        json.dump(sea_level_diffs, diffs_file)
    metadata_path = os.path.join(out_directory, "metadata.txt")
    # create line graph
    out_directory = "../output/RMS/line_graph_2010_2020_simulated_data_remove_stations_not_present_in_real_data/"
    grouped_stations = group_stations(working_stations, metadata_path)
    result_graph = create_line_graph(grouped_stations, sea_level_differences, working_stations, metadata_path)
    # write metadata to file
    with open(os.path.join(out_directory, "metadata.txt"), "w") as metadata_file:
        metadata_file.write(f"Number of components in graph: {networkx.number_connected_components(result_graph)}\n")
    node_data = {}
    for station in all_stations:
        node_data[station.id] = [station.longitude, station.latitude]
    with open(os.path.join(out_directory, "node_data.json"), "w") as json_file:
        json.dump(node_data, json_file)
    result_gdf = gdf_from_graph(result_graph)
    networkx.write_edgelist(result_graph, os.path.join(out_directory, "line_graph.csv"),
                            delimiter=";", data=False)
    present_stations_gdf = measuring_stations_gdf[measuring_stations_gdf["id"].isin(result_graph.nodes)]
    plot_line_graph(land_gdf, out_directory, present_stations_gdf, result_gdf, "line_graph")


def start(station_path: str, land_path: str, difference_file_path: str, difference_file_name: str,
          simulated_sea_level_differences_path: str, output_path: str, graph_per_time_steps: bool,
          reduce_graph_per_timestep: bool):
    pass
    # # ------------------------------------------------------------------------
    # # calculate the line graph based on the simulated data, but only for the stations that are present in the
    # # timespan between 2010 and 2020
    # # input for simulated data
    # input_path = "../../time_series_difference_calculator/output/RMS/simulated_data_cleaned/2010_2020/differences
    # .json"
    # sea_level_differences = read_sea_level_differences(input_path)
    # # # input path for the real data
    # # input_path_real_data = (
    # #     "../../time_series_difference_calculator/output/RMS/timesteps_remove_011_flagged_data/2010_2020/differences"
    # #     ".json")
    # real_sea_level_diffs = read_sea_level_differences(input_path_real_data)
    # # # remove stations that are not present in the timespan
    #
    # create_simulated_line_graph_based_on_real_data(all_stations, sea_level_differences, real_sea_level_diffs,
    # land_gdf,
    #                                                measuring_stations_gdf)


def create_base_graph(mae: bool, mean_center_and_detrending: bool, output_dir: str, rms: bool,
                      stations: {int: tide_gauge_station.TideGaugeStation}, land_for_plotting: str):
    """
    Create the base graph for reducing the line graph for each time step
    :param mae:
    :param mean_center_and_detrending:
    :param output_dir:
    :param rms:
    :param stations:
    :param land_for_plotting:
    :return:
    """
    if not os.path.exists(os.path.join(output_dir, "line_graph_complete_timespan.csv")):
        # calculate differences and sea level line graph for all stations
        if mean_center_and_detrending:
            all_stations = tide_gauge_station.detrend_and_mean_center_timeseries(stations)
        else:
            all_stations = stations
            for station in all_stations.values():
                station.timeseries_detrended_normalized = station.timeseries.copy()
        if not os.path.exists(os.path.join(output_dir, "difference.txt")):
            sea_level_differences_complete_timespan, sea_level_difference_percentage = (
                timeseries_difference.calculate_time_series_difference(output_dir, all_stations, mae, rms))
            timeseries_difference.save_differences_to_file(sea_level_differences_complete_timespan, output_dir,
                                                           "difference.txt")
            timeseries_difference.save_differences_to_file(sea_level_difference_percentage, output_dir,
                                                           "percentage.txt")

        sea_level_differences_complete_timespan = timeseries_difference.read_differences_from_file(output_dir,
                                                                                                   "difference.txt")
        sea_level_line_graph_complete_timespan = construct_line_graph_for_current_timestep(
            os.path.join(output_dir, "metadata.txt"), sea_level_differences_complete_timespan,
            list(all_stations.values()))
        save_and_plot_line_graph(land_for_plotting, "line_graph_complete_timespan", output_dir,
                                 sea_level_line_graph_complete_timespan, all_stations)
    else:
        sea_level_line_graph_complete_timespan = read_line_graph(
            os.path.join(output_dir, "line_graph_complete_timespan.csv"),
            os.path.join(output_dir, "node_data.txt"))

    return sea_level_line_graph_complete_timespan
