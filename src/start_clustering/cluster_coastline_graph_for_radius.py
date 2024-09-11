import json
import os

from loguru import logger
from tqdm import tqdm

from src.inner import tide_gauge_station, sea_level_line_graph, timeseries_difference, line_graph_clustering


def start(list_of_radii: [float], coastline_line_graph_path: str, node_path: str,
          stations: {int: tide_gauge_station.TideGaugeStation}, output_dir: str, mean_center_and_detrending: bool,
          rms: bool, mae: bool):
    """
    Calculates one solution for each radius in the list_of_radii given a coastline line graph and a list of stations
    :param node_path:
    :param list_of_radii:
    :param time_steps:
    :param coastline_line_graph_path:
    :param stations:
    :param output_dir:
    :param mean_center_and_detrending:
    :param rms:
    :param mae:
    :param percentage:
    :param land_for_plotting:
    :return:
    """
    logger.info("Start connected clustering with coastline line graph for radii")
    coastline_line_graph = sea_level_line_graph.read_line_graph(coastline_line_graph_path, node_path)
    # calculate time series difference for the complete time span
    if not os.path.exists(os.path.join(output_dir, "difference.txt")):
        logger.info("Calculating difference between pairs of time series for complete time span")
        if mean_center_and_detrending:
            stations = tide_gauge_station.detrend_and_mean_center_timeseries(stations)
        else:
            for station in stations.values():
                station.timeseries_detrended_normalized = station.timeseries.copy()
        sea_level_diff, sea_level_percentage = timeseries_difference.calculate_time_series_difference(output_dir,
                                                                                                      stations, mae,
                                                                                                      rms)
    else:  # read time series difference from file
        sea_level_diff = timeseries_difference.read_differences_from_file(output_dir, "difference.txt")
        sea_level_percentage = timeseries_difference.read_differences_from_file(output_dir, "percentage.txt")
    # calculate connected clustering for the given radii
    minimum_overlap_wanted = False
    overlap = 0
    logger.info(f"Start connected clustering for radii {list_of_radii}")
    for radius in tqdm(list_of_radii):
        number_of_centers, all_center_nodes = line_graph_clustering.compute_clustering_for_given_radius(
            coastline_line_graph, radius, sea_level_diff, minimum_overlap_wanted, overlap)
        solution = {}
        for center in all_center_nodes.keys():
            solution[center] = list(all_center_nodes[center].nodes())
        with open(os.path.join(output_dir, f"solution_{radius}.json"), "w") as file:
            json.dump(solution, file)
        with open(os.path.join(output_dir, f"clustering_info_{radius}.txt"), "w") as file:
            file.write("cluster radius: " + str(radius) + "\n")
            file.write(f"number of centers: {len(solution.keys())}\n")
            for cluster in solution.keys():
                file.write(f"{len(solution[cluster])}; ")
            file.write(f"\nAverage cluster size {len(stations) / number_of_centers}\n")
            file.write("\n")
    return
