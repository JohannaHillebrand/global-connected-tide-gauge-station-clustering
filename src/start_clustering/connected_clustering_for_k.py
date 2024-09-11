import json
import os.path

from loguru import logger
from tqdm import tqdm

from src.inner import sea_level_line_graph, line_graph_clustering
from src.inner import tide_gauge_station, timeseries_difference
from src.inner.sea_level_line_graph import create_base_graph
from src.inner.timeseries_difference import calculate_time_series_difference


def start(list_of_k: [int], time_steps: [],
          stations: {int: tide_gauge_station.TideGaugeStation}, output_dir: str,
          mean_center_and_detrending: bool, rms: bool, mae: bool, percentage: int, reduce_graph_per_time_step: bool,
          land_for_plotting: str):
    """
    connected clustering for k clusters
    :return:
    """
    if reduce_graph_per_time_step:
        logger.info("Creating base graph for complete timespan")
        sea_level_line_graph_complete_timespan = create_base_graph(mae, mean_center_and_detrending, output_dir, rms,
                                                                   stations, land_for_plotting)

    logger.info(f"Start connected clustering for k = {list_of_k} and time steps {time_steps}")
    for time_step in tqdm(time_steps):
        # logger.info(f"Start for time step {time_step}")
        current_output_dir = os.path.join(output_dir, f"{time_step[0]}_{time_step[1]}")
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        start_year = time_step[0]
        end_year = time_step[1]
        filtered_stations = tide_gauge_station.filter_stations_for_time_step(stations, start_year, end_year)
        if mean_center_and_detrending:
            filtered_stations = tide_gauge_station.detrend_and_mean_center_timeseries(filtered_stations)
        else:
            for station in filtered_stations.values():
                station.timeseries_detrended_normalized = station.timeseries.copy()
        if not os.path.exists(os.path.join(current_output_dir, "difference.txt")):
            # calculate timeseries difference if not present for this timestep
            # logger.info(f"Calculating difference between pairs of time series for time step {time_step}")
            current_difference, current_percentage = calculate_time_series_difference(current_output_dir,
                                                                                      filtered_stations, mae,
                                                                                      rms)
        else:
            # read timeseries difference from file
            current_difference = timeseries_difference.read_differences_from_file(current_output_dir, "difference.txt")
            current_percentage = timeseries_difference.read_differences_from_file(current_output_dir, "percentage.txt")
        # calculate line graph if not present for this timestep
        if not os.path.exists(os.path.join(current_output_dir, f"line_graph_{time_step}")):
            graph_metadata_path = os.path.join(current_output_dir, "graph_metadata.txt")
            if not reduce_graph_per_time_step:
                # logger.info(f"Calculating line graph for time step {time_step}")
                # calculate line graph per timestep
                line_graph = sea_level_line_graph.construct_line_graph_for_current_timestep(graph_metadata_path,
                                                                                            current_difference,
                                                                                            list(
                                                                                                filtered_stations.values()))
                sea_level_line_graph.save_and_plot_line_graph(land_for_plotting, f"line_graph_{time_step}",
                                                              current_output_dir, line_graph, filtered_stations)

            else:
                # reduce line graph per timestep
                line_graph = sea_level_line_graph.reduce_line_graph(sea_level_line_graph_complete_timespan,
                                                                    list(filtered_stations.values()),
                                                                    os.path.join(current_output_dir,
                                                                                 "graph_metadata.txt"))
                sea_level_line_graph.save_and_plot_line_graph(land_for_plotting, f"line_graph_{time_step}",
                                                              current_output_dir, line_graph, filtered_stations)
        else:
            # read line graph from file
            line_graph = sea_level_line_graph.read_line_graph(os.path.join(current_output_dir, "line_graph.csv"),
                                                              os.path.join(current_output_dir, "node_data.txt"))
        input_for_clustering = []
        # logger.info(f"Calculating clustering for time step {time_step} for k = {list_of_k}")
        if percentage:
            minimum_overlap_wanted = True
            overlap = percentage
        else:
            minimum_overlap_wanted = False
            overlap = 0
        for k in list_of_k:
            result = line_graph_clustering.cluster_for_k(k, line_graph, current_difference,
                                                         minimum_overlap_wanted, overlap)
            if result is not None:
                clustering, max_radius, elapsed_time, number_iterations = result
                # save clustering
                with open(f"{current_output_dir}/clustering_info_{k}.txt", "w") as file:
                    file.write(f"number of centers: {len(clustering.keys())}\n")
                    file.write("cluster radius: " + str(max_radius) + "\n")
                    file.write("elapsed time: " + str(elapsed_time) + "\n")
                    file.write("number of iterations: " + str(number_iterations) + "\n")
                with open(f"{current_output_dir}/solution_{k}.json", "w") as file:
                    json.dump(clustering, file)

    return
