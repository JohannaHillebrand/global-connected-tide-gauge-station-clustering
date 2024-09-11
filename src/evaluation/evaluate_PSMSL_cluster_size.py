import json
import os

import numpy

from src.inner import tide_gauge_station


def iterate_over_radii(current_output_path: str, all_radii: [float], all_time_steps: [(int, int)],
                       stations: {int: tide_gauge_station.TideGaugeStation}, current_clustering_path: str):
    """
    Iterate over all radii and time steps and evaluate the cluster size
    :param current_output_path:
    :param all_radii:
    :param all_time_steps:
    :param stations:
    :param current_clustering_path:
    :return:
    """
    evaluation_path = os.path.join(current_output_path, "evaluation.txt")
    with open(evaluation_path, "w") as eval_file:
        eval_file.write(f"Evaluating: {current_output_path}\n\n")
    average_number_clusters_per_radius = {}
    median_number_clusters_per_radius = {}
    max_number_clusters_per_radius = {}
    min_number_clusters_per_radius = {}
    average_cluster_size_per_radius = {}
    for radius in all_radii:
        center_reoccurrence = {}
        number_of_clusters = []
        avg_cluster_size = 0
        number_of_clusters_per_number_of_stations = []
        for time_step in all_time_steps:
            current_solution_path = os.path.join(current_clustering_path, f"{time_step[0]}_{time_step[1]}",
                                                 f"solution_{radius}.json")
            with open(current_solution_path, "r") as input_file:
                current_solution = json.load(input_file)
            for center in current_solution.keys():
                if center not in center_reoccurrence:
                    center_reoccurrence[center] = 0
                center_reoccurrence[center] += 1
            current_number_of_clusters = len(current_solution.keys())
            current_avg_cluster_size = sum(
                [len(cluster) for cluster in current_solution.values()]) / current_number_of_clusters
            number_of_clusters_per_number_of_stations.append((sum(
                [len(cluster) for cluster in current_solution.values()]) / len(current_solution)))
            number_of_clusters.append(current_number_of_clusters)
            avg_cluster_size += current_avg_cluster_size
        average_cluster_size_per_radius[radius] = avg_cluster_size / len(all_time_steps)
        average_number_clusters_per_radius[radius] = sum(number_of_clusters) / len(all_time_steps)
        median_number_clusters_per_radius[radius] = numpy.median(number_of_clusters)
        max_number_clusters_per_radius[radius] = numpy.max(number_of_clusters)
        min_number_clusters_per_radius[radius] = numpy.min(number_of_clusters)
        with open(evaluation_path, "a") as eval_file:
            eval_file.write(f"Radius: {radius} \n")
            eval_file.write(f"Average Cluster size: {round(avg_cluster_size / len(all_time_steps), 2)}\n")
            eval_file.write(f"Average number of clusters: {round(sum(number_of_clusters) / len(all_time_steps), 2)}\n")
            eval_file.write(f"Median number of clusters: {round(numpy.median(number_of_clusters), 2)}\n")
            eval_file.write(f"Max number of clusters: {round(numpy.max(number_of_clusters), 2)}\n")
            eval_file.write(f"Min number of clusters: {round(numpy.min(number_of_clusters), 2)}\n")
            eval_file.write(
                f"Average cluster size per number of stations: "
                f"{round(sum(number_of_clusters_per_number_of_stations) / len(time_steps), 2)}\n")
            eval_file.write(
                f"Center reoccurrence: "
                f"{sum([reoccurrence for reoccurrence in center_reoccurrence.values() if reoccurrence != 0])}\n")
            eval_file.write(
                f"Center reoccurrence relative to the number of centers: "
                f"{round(sum([reoccurrence for reoccurrence in center_reoccurrence.values() if reoccurrence != 0]) / sum([(frequency + 1) for center, frequency in center_reoccurrence.items()]), 2)} \n")
            eval_file.write(
                f"center reoccurrence on average: {round(numpy.mean(list(center_reoccurrence.values())), 2)}\n")
            eval_file.write(f"\n\n")

    pass


def start(current_output_path: str, all_radii: [float], all_time_steps: [(int, int)],
          stations: {int: tide_gauge_station.TideGaugeStation}, current_clustering_path: str):
    """
    Evaluate the cluster size for different radii for the PSMSL dataset
    :param current_output_path:
    :param all_radii:
    :param all_time_steps:
    :param stations:
    :param current_clustering_path:
    :return:
    """
    iterate_over_radii(current_output_path, all_radii, all_time_steps, stations, current_clustering_path)
    pass


if __name__ == "__main__":
    """
    Evaluate the cluster size for different radii for the PSMSL dataset
    """
    clustering_path = "../../output/time_steps/"
    output_path = clustering_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # PSMSL_path = "../../data/rlr_monthly/filelist.txt"
    PSMSL_path = "../../data/Meeresdaten_simuliert/filelist.txt"
    station_data = tide_gauge_station.read_and_create_stations(PSMSL_path, os.path.join(output_path, "metadata.txt"))
    time_steps = [(i, i + 10) for i in range(1958, 2028, 10)]
    # radii = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    radii = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    start(output_path, radii, time_steps, station_data, clustering_path)
