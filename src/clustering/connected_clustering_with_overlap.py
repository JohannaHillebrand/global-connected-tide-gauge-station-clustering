import json
import os

from loguru import logger
from tqdm import tqdm

from src.inner import tide_gauge_station, timeseries_difference, sea_level_line_graph, line_graph_clustering


def start(list_of_radii: [float], time_steps: [(int, int)], stations: {int: tide_gauge_station.TideGaugeStation},
          output_dir: str, mean_center_and_detrending: bool, rms: bool, mae: bool, percentage: int,
          reduce_graph_per_time_step: bool, land_for_plotting: str, overlap: int):
    """
    Calculate the connected clustering for a given list of radii
    :param overlap:
    :param list_of_radii:
    :param time_steps:
    :param stations:
    :param output_dir:
    :param mean_center_and_detrending:
    :param rms:
    :param mae:
    :param percentage:
    :param reduce_graph_per_time_step:
    :param land_for_plotting:
    :return:
    """
    if reduce_graph_per_time_step:
        logger.info("Creating base graph for complete timespan")
        sea_level_line_graph_complete_timespan = sea_level_line_graph.create_base_graph(mae, mean_center_and_detrending,
                                                                                        output_dir, rms, stations,
                                                                                        land_for_plotting)
    all_radii_solutionsize_path = os.path.join(output_dir, "all_radii_solutionsize.csv")
    logger.info(f"Start connected clustering for radii {list_of_radii} and time steps {time_steps}")
    for time_step in tqdm(time_steps):
        # logger.info(f"Start for time step {time_step}")
        with open(all_radii_solutionsize_path, "a") as file:
            file.write(f"Time step: {time_step}\n")
        current_output_dir = f"{output_dir}/{time_step[0]}_{time_step[1]}"
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
        if not os.path.exists(f"{current_output_dir}/difference.txt"):
            # calculate timeseries difference if not present for this timestep
            # logger.info(f"Calculating difference between pairs of time series for time step {time_step}")
            current_difference, current_percentage = timeseries_difference.calculate_time_series_difference(
                current_output_dir, filtered_stations, mae, rms)
        else:
            # read timeseries difference from file
            current_difference = timeseries_difference.read_differences_from_file(current_output_dir, "difference.txt")
            current_percentage = timeseries_difference.read_differences_from_file(current_output_dir, "percentage.txt")
        # calculate line graph if not present for this timestep
        if not os.path.exists(f"{current_output_dir}/line_graph_{time_step}.csv"):
            graph_metadata_path = f"{current_output_dir}/graph_metadata.txt"
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
                # logger.info(f"Reducing line graph for time step {time_step}")
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
        # logger.info(f"Calculating connected clustering for time step {time_step} for radii {list_of_radii}")
        if percentage:
            minimum_overlap_wanted = True
            overlap = percentage
            diff_with_percentage = {}
            for key in current_difference.keys():
                for second_key in current_difference[key].keys():
                    diff_with_percentage[key][second_key] = (
                        current_difference[key][second_key], current_percentage[key][second_key])
            current_difference = diff_with_percentage

        else:
            minimum_overlap_wanted = False
            overlap = 0
        for radius in list_of_radii:
            current_number_of_centers, all_center_nodes = line_graph_clustering.compute_clustering_for_given_radius(
                line_graph, radius, current_difference,
                minimum_overlap_wanted, overlap)
            solution = {}
            for center in all_center_nodes.keys():
                solution[center] = list(all_center_nodes[center].nodes())
            with open(f"{current_output_dir}/solution_{radius}.json", "w") as file:
                json.dump(solution, file)
            with open(f"{current_output_dir}/info_solution_{radius}.txt", "w") as file:
                file.write(f"Radius: {radius}\n")
                file.write(f"Number of clusters: {current_number_of_centers}\n")
                for cluster in solution.keys():
                    file.write(f"{len(solution[cluster])}; ")
                file.write(f"\nAverage cluster size {len(filtered_stations) / current_number_of_centers}\n")
                file.write("\n")
            with open(all_radii_solutionsize_path, "a") as file:
                file.write(
                    f"{radius};{current_number_of_centers};{len(filtered_stations) / current_number_of_centers}\n")
    return
