import json
import os
import random

from loguru import logger

import src.inner.tide_gauge_station
from src.evaluation import evaluate_clustering_PSMSL_global_sea_level
from src.inner import timeseries_difference, plot


def determine_sample_sizes(all_radii: [float], all_time_steps: [str], current_clustering_directory: str,
                           stations: {int: timeseries_difference.TideGaugeStation}):
    """
    Calculate the highest number of clusters for each radius over all time steps, this is the size of the random sample
    for each radius
    :param stations:
    :param all_radii:
    :param all_time_steps:
    :param current_clustering_directory:
    :return:
    """
    logger.info("Calculating average number of clusters for each radius")
    cluster_size_per_radius = {}
    for current_radius in all_radii:
        if current_radius == 0.0:
            cluster_size_per_radius[current_radius] = len(stations.keys())
            continue
        # avg_number_of_clusters = 0
        largest_number_of_clusters = 0
        for current_time_step in all_time_steps:
            input_directory = os.path.join(current_clustering_directory,
                                           f"{current_time_step}/solution{current_radius}.json")
            with open(input_directory, "r") as f:
                current_clustering = json.load(f)
            current_number_of_clusters = len(current_clustering.keys())
            if current_number_of_clusters > largest_number_of_clusters:
                largest_number_of_clusters = current_number_of_clusters
            # avg_number_of_clusters += current_number_of_clusters
        # avg_number_of_clusters = avg_number_of_clusters / len(all_time_steps)
        cluster_size_per_radius[current_radius] = int(largest_number_of_clusters)
    return cluster_size_per_radius


def draw_random_solution(cluster_size_per_radius: dict, stations: {}, all_time_steps: [str], output_dir: str):
    """
    Create a random sample of the calculated size for each radius
    :param cluster_size_per_radius:
    :param stations:
    :param all_time_steps:
    :param output_dir:
    :return:
    """
    id_list = list(stations.keys())
    for radius in cluster_size_per_radius.keys():
        current_random_clustering = {}
        random_sample = random.sample(id_list, cluster_size_per_radius[radius])
        for station_id in random_sample:
            current_random_clustering[station_id] = []
        for time_step in all_time_steps:
            if not os.path.exists(os.path.join(output_dir, time_step)):
                os.makedirs(os.path.join(output_dir, time_step))
            with open(os.path.join(output_dir, f"{time_step}/solution{radius}.json"), "w") as f:
                json.dump(current_random_clustering, f)


def draw_random_solution_equal_per_hemisphere(clustering_size_per_radius, stations, all_time_steps, out_directory):
    """
    Create a random sample of the calculated size for each radius, with an equal amount of stations per hemisphere
    :param clustering_size_per_radius: 
    :param stations: 
    :param all_time_steps: 
    :param out_directory: 
    :return: 
    """
    northern_stations = {}
    southern_stations = {}
    for station_id, station in stations.items():
        if station.latitude > 0:
            northern_stations[station_id] = station
        else:
            southern_stations[station_id] = station
    logger.info(
        f"northern stations: {len(northern_stations.keys())}, southern stations: {len(southern_stations.keys())}")
    for radius in clustering_size_per_radius.keys():
        current_random_clustering = {}
        if len(northern_stations.keys()) < clustering_size_per_radius[radius] / 2:
            northern_sample = random.sample(list(northern_stations.keys()), len(northern_stations.keys()))
            southern_sample = random.sample(list(southern_stations.keys()),
                                            clustering_size_per_radius[radius] - len(northern_stations.keys()))
            logger.warning(
                f"An equal distribution of randomly drawn stations per hemisphere is not possible for radius "
                f"{radius}mm. There are too few northern stations ({len(northern_stations)}).")

        elif len(southern_stations.keys()) < clustering_size_per_radius[radius] / 2:
            southern_sample = random.sample(list(southern_stations.keys()), len(southern_stations.keys()))
            northern_sample = random.sample(list(northern_stations.keys()),
                                            clustering_size_per_radius[radius] - len(southern_stations.keys()))
            logger.warning(
                f"An equal distribution of randomly drawn stations per hemisphere is not possible for radius "
                f"{radius}mm. There are too few southern stations ({len(southern_stations)}).")
        else:
            northern_sample = random.sample(list(northern_stations.keys()), int(clustering_size_per_radius[radius] / 2))
            southern_sample = random.sample(list(southern_stations.keys()), int(clustering_size_per_radius[radius] / 2))
        for station_id in northern_sample:
            current_random_clustering[station_id] = []
        for station_id in southern_sample:
            current_random_clustering[station_id] = []
        # logger.info(f"number of stations for radius {radius}: {len(current_random_clustering.keys())}")
        for time_step in all_time_steps:
            if not os.path.exists(os.path.join(out_directory, time_step)):
                os.makedirs(os.path.join(out_directory, time_step))
            with open(os.path.join(out_directory, f"{time_step}/solution{radius}.json"), "w") as f:
                json.dump(current_random_clustering, f)


if __name__ == "__main__":
    station_path = "../data/rlr_monthly/filelist.txt"
    out_dir = "../output/../output/random_sample_for_voronoi_section_clustering/v2/"
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.txt")
    altimetry_data_path = "../data/global_sea_level_altimetry/gmsl_2023rel2_seasons_retained.txt"
    # radii for the clustering in mm
    radii_for_reading = [0.0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    # "radii" for section clustering - here we do not really have radii, but the number of centers
    radii = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    time_steps = ["1992_2002", "2002_2012", "2012_2022", "2022_2032"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # read in stations and create a random sample of the calculated size for each radius
    all_stations = src.inner.tide_gauge_station.read_and_create_stations(station_path, metadata_path)
    # read all clustering results for every radius and calculate the average number of clusters for each radius over
    # all time steps
    clustering_directory = "../output/Voronoi"
    sample_size_per_radius = determine_sample_sizes(radii, time_steps, clustering_directory, all_stations)
    logger.info(f"Total number of stations: {len(all_stations)}, Sample size per radius: {sample_size_per_radius}")

    # draw a random sample of stations of the same size as the average number of clusters for each radius
    # only use this if a new random sample is needed
    # draw_random_solution(sample_size_per_radius, all_stations, time_steps, out_dir)

    # random sample equal amount per hemisphere
    # use this if an equal amount of stations per hemisphere is needed
    out_dir = "../output/random_sample_for_voronoi_section_clustering/v2/equal_per_hemisphere"
    # draw_random_solution_equal_per_hemisphere(sample_size_per_radius, all_stations, time_steps, out_dir)
    northern = 0
    southern = 0
    for station_id, station in all_stations.items():
        if station.latitude > 0:
            northern += 1
        else:
            southern += 1
    logger.info(f"Northern stations: {northern}, Southern stations: {southern}")

    # read in altimetry data and evaluate the random sample
    altimetry_data = evaluate_clustering_PSMSL_global_sea_level.read_altimetry_data(altimetry_data_path)
    logger.info("Evaluating random sample")
    rms_all, timeline_all = evaluate_clustering_PSMSL_global_sea_level.calculate_rms_all_stations(all_stations, radii,
                                                                                                  altimetry_data,
                                                                                                  out_dir)
    rms_clustered, rms_per_number_of_centers = evaluate_clustering_PSMSL_global_sea_level.evaluate_clustering(radii,
                                                                                                              time_steps,
                                                                                                              altimetry_data,
                                                                                                              all_stations,
                                                                                                              timeline_all,
                                                                                                              out_dir,
                                                                                                              clustering_directory)
    evaluate_clustering_PSMSL_global_sea_level.plot_save_rms(rms_all, rms_per_number_of_centers, out_dir, "teal")
    rms_per_number_of_centers = {}
    for radius in rms_clustered.keys():
        current_size = sample_size_per_radius[radius]
        rms_per_number_of_centers[current_size] = rms_clustered[radius]

    out_dir = "../output/random_sample_for_voronoi_section_clustering/v2"
    # rms all per number of centers
    rms_all_stations_per_number_of_centers = {}
    for number_of_centers in rms_per_number_of_centers.keys():
        rms_all_stations_per_number_of_centers[number_of_centers] = rms_all[50]
    rms_all_stations_per_radius = {}
    for radius in rms_clustered.keys():
        rms_all_stations_per_radius[radius] = rms_all[50]

    # calculate the difference between the randomly selected stations and the global sea level for the time steps
    random_selection_path = "../output/random_sample/v2"
    random_output_path = "../output/random_sample_for_voronoi_section_clustering/v2"
    os.makedirs(random_output_path, exist_ok=True)
    rms_random_sample, rms_random_sample_per_number_of_centers = (
        evaluate_clustering_PSMSL_global_sea_level.evaluate_clustering(
            radii_for_reading,
            time_steps,
            altimetry_data,
            all_stations,
            timeline_all,
            random_output_path,
            random_selection_path))
    # calculate the difference between the random-equal distributed stations and the global sea level for the time steps
    random_equal_distribution_path = "../output/random_sample/v2/equal_per_hemisphere"
    random_equal_distribution_output_path = \
        "../output/random_sample_for_voronoi_section_clustering/v2/equal_per_hemisphere/"
    os.makedirs(random_equal_distribution_output_path, exist_ok=True)
    rms_random_equal_distribution, rms_random_equal_distribution_per_number_of_centers = (
        evaluate_clustering_PSMSL_global_sea_level.evaluate_clustering(
            radii_for_reading,
            time_steps,
            altimetry_data,
            all_stations,
            timeline_all,
            random_equal_distribution_output_path,
            random_equal_distribution_path))

    # remove the random solutions that are larger than the clustered solution
    to_remove = []
    for number_of_centers in rms_random_sample_per_number_of_centers.keys():
        if number_of_centers > 700:
            to_remove.append(number_of_centers)
    for station_id in to_remove:
        rms_random_sample_per_number_of_centers.pop(station_id)
    to_remove = []
    for number_of_centers in rms_random_equal_distribution_per_number_of_centers.keys():
        if number_of_centers > 700:
            to_remove.append(number_of_centers)
    for station_id in to_remove:
        rms_random_equal_distribution_per_number_of_centers.pop(station_id)

    # per number or stations
    rms_clustering_color = "teal"
    rms_all_stations_colors = "firebrick"
    # plot_save_rms(rms_all_stations_per_number_of_centers, rms_clustered, output_path, rms_clustering_color)
    x_label = "Number of centers"
    plot.plot_rmse_graph([(rms_per_number_of_centers, "RMSE clustered centers", rms_clustering_color),
                          (rms_all_stations_per_number_of_centers, "RMSE all stations", rms_all_stations_colors)],
                         "rmse_clustered_all_no_clusters",
                         out_dir, x_label)
    x_label = "Radius"
    plot.plot_rmse_graph([(rms_clustered, "RMSE clustered centers", rms_clustering_color),
                          (rms_all_stations_per_radius, "RMSE all stations", rms_all_stations_colors)],
                         "rmse_clustered_all_radius",
                         out_dir, x_label)

    random_sample_color = "purple"
    x_label = "Number of centers"
    plot.plot_rmse_graph([(rms_per_number_of_centers, "RMSE clustered centers", rms_clustering_color),
                          (rms_random_sample_per_number_of_centers, "RMSE random sample", random_sample_color),
                          (rms_all_stations_per_number_of_centers, "RMSE all stations", rms_all_stations_colors)],
                         "rms_clustered_random", out_dir, x_label)
    plot.plot_rmse_graph([(rms_random_sample_per_number_of_centers, "RMSE random sample", random_sample_color),
                          (rms_all_stations_per_number_of_centers, "RMSE all stations", rms_all_stations_colors)],
                         "rms_random", out_dir, x_label)
    x_label = "Radius"
    plot.plot_rmse_graph([(rms_clustered, "RMSE clustered centers", rms_clustering_color),
                          (rms_random_sample, "RMSE random sample", random_sample_color),
                          (rms_all_stations_per_radius, "RMSE all stations", rms_all_stations_colors)],
                         "rms_clustered_random_radius", out_dir, x_label)
    plot.plot_rmse_graph([(rms_random_sample, "RMSE random sample", random_sample_color),
                          (rms_all_stations_per_radius, "RMSE all stations", rms_all_stations_colors)],
                         "rms_random_radius", out_dir, x_label)

    random_equal_distribution_color = "goldenrod"
    x_label = "Number of centers"
    plot.plot_rmse_graph([(rms_per_number_of_centers, "RMSE clustered centers", rms_clustering_color),
                          (rms_random_sample_per_number_of_centers, "RMSE random sample", random_sample_color),
                          (rms_random_equal_distribution_per_number_of_centers, "RMSE random equal distribution",
                           random_equal_distribution_color),
                          (rms_all_stations_per_number_of_centers, "RMSE all stations", rms_all_stations_colors)],
                         "rms_clustered_random_equal",
                         out_dir, x_label)
    plot.plot_rmse_graph([(rms_random_equal_distribution_per_number_of_centers, "RMSE random equal distribution",
                           random_equal_distribution_color),
                          (rms_all_stations_per_number_of_centers, "RMSE all stations", rms_all_stations_colors)],
                         "rms_equal", out_dir, x_label)
    x_label = "Radius"
    plot.plot_rmse_graph([(rms_clustered, "RMSE clustered centers", rms_clustering_color),
                          (rms_random_sample, "RMSE random sample", random_sample_color),
                          (rms_random_equal_distribution, "RMSE random equal distribution",
                           random_equal_distribution_color),
                          (rms_all_stations_per_radius, "RMSE all stations", rms_all_stations_colors)],
                         "rms_clustered_random_equal_radius",
                         out_dir, x_label)
    plot.plot_rmse_graph([(rms_random_equal_distribution, "RMSE random equal distribution",
                           random_equal_distribution_color),
                          (rms_all_stations_per_radius, "RMSE all stations", rms_all_stations_colors)],
                         "rms_equal_radius", out_dir, x_label)
    x_label = "Number of centers"
    plot.plot_rmse_graph([(rms_per_number_of_centers, "RMSE clustered centers", rms_clustering_color),
                          (rms_all_stations_per_number_of_centers, "RMSE all stations", rms_all_stations_colors), (
                              rms_random_equal_distribution_per_number_of_centers, "RMSE random equal distribution",
                              random_equal_distribution_color)],
                         "rmse_clustered_all_equal_dist_no_clusters", out_dir, x_label)
