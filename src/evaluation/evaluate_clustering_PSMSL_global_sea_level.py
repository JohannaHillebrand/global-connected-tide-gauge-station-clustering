import json
import os

import numpy

import src.inner.tide_gauge_station
from src.inner import timeseries_difference, plot


def find_closest_date(altimetry: {float: float}, psmsl: [float]):
    """
    Find the closest date in the altimetry data to the PSMSL data
    :param psmsl:
    :param altimetry:
    :return:
    """
    closest_dates = {}
    for date in psmsl:
        if date >= 1992.9:
            closest_date = float("inf")
            for altimetry_date in altimetry.keys():
                if abs(date - altimetry_date) < abs(date - closest_date):
                    closest_date = altimetry_date
            closest_dates[date] = closest_date
    return closest_dates


def read_altimetry_data(altimetry_path: str):
    """
    Read altimetry data
    :param altimetry_path:
    :return:
    """
    altimetry = {}
    with open(altimetry_path) as out_file:
        for line in out_file:
            line = line.split()
            current_date = float(line[0].strip())
            sea_level = float(line[1].strip())
            altimetry[current_date] = sea_level
    # mean center the values
    current_sum = 0
    for value in altimetry.values():
        current_sum += value
    mean = current_sum / len(altimetry)
    altimetry_mean_centered = {}
    for current_date in altimetry.keys():
        sea_level = altimetry[current_date]
        altimetry_mean_centered[current_date] = round(sea_level - mean, 4)
    return altimetry_mean_centered


def calculate_rms(altimetry: {float: float}, current_clustered_solution: {float: float}, date_mapping: {float: float}):
    """
    Calculate the root-mean-square error
    :return:
    """
    sorted_current_clustered_solution = dict(sorted(current_clustered_solution.items()))
    sorted_global_mean_sea_level = dict(sorted(altimetry.items()))
    # calculate the root mean square error
    current_sum = 0
    for date in sorted_current_clustered_solution.keys():
        current_sum += (sorted_current_clustered_solution[date] - sorted_global_mean_sea_level[date_mapping[date]]) ** 2
    rms = numpy.sqrt(current_sum / len(sorted_current_clustered_solution))
    return rms


def extract_clustering_values(stations: {str: src.inner.tide_gauge_station.TideGaugeStation},
                              current_clustering: {str: int},
                              start_year: float, end_year: float, sea_level_diffs: {int: {int: float}}):
    """
    Extract the average clustering
    :return:
    """
    collect_clustering = {}
    max_difference = 0
    all_differences = []
    for center_id in current_clustering.keys():
        current_station = stations[int(center_id)]
        current_timeseries = current_station.timeseries
        for date in current_timeseries.keys():
            if current_timeseries[date] == -99999:
                continue
            if start_year <= date < end_year:
                if date >= 1992.9:
                    if date not in collect_clustering.keys():
                        collect_clustering[date] = [current_timeseries[date]]
                    else:
                        collect_clustering[date].append(current_timeseries[date])
        # check difference between center and associated stations
        for second_station_id in current_clustering[center_id]:
            if int(center_id) == int(second_station_id):
                continue
            try:
                diff = sea_level_diffs[int(center_id)][int(second_station_id)]
                all_differences.append(diff)
                if diff > max_difference:
                    max_difference = diff
            except KeyError:
                print(f"KeyError: {center_id} {second_station_id}")
    return collect_clustering, max_difference, all_differences


def evaluate_clustering(all_radii: [float], all_time_steps: [str], current_altimetry_data: {float: float},
                        all_station_data: {int: timeseries_difference.TideGaugeStation},
                        timeline_all_stations: {float: float},
                        output_directory: str, input_directory: str):
    """
    Evaluate the clustering
    :param timeline_all_stations:
    :param input_directory:
    :param all_radii:
    :param all_time_steps:
    :param current_altimetry_data:
    :param all_station_data:
    :param output_directory:
    :return:
    """
    rms_per_radius = {}
    rms_per_number_of_clusters = {}
    clustered_values = {}
    avg_diff_per_radius = {}
    max_diff_per_radius = {}
    for radius in all_radii:
        avg_solution_size = 0
        max_diff = 0
        all_diffs = []
        for time_step in all_time_steps:
            start_year = float(time_step.split("_")[0])
            end_year = float(time_step.split("_")[1])
            file_path = (f"{input_directory}/{time_step}/solution_"
                         f"{radius}.json")
            if not os.path.exists(file_path):
                file_path = (f"{input_directory}/{time_step}/solution_sealevel"
                             f"{radius}.json")
            if not os.path.exists(file_path):
                file_path = (f"{input_directory}/{time_step}/solution"
                             f"{radius}.json")
            sea_level_diffs = timeseries_difference.read_differences_from_file(os.path.join(input_directory, time_step),
                                                                               "difference.txt")
            with open(file_path) as file:
                current_solution = json.load(file)
                current_clustered_values, current_max_diff, current_all_diffs = extract_clustering_values(
                    all_station_data, current_solution, start_year, end_year, sea_level_diffs)
                if current_max_diff > max_diff:
                    max_diff = current_max_diff
                all_diffs.extend(current_all_diffs)
                clustered_values.update(current_clustered_values)
                avg_solution_size += len(current_solution)
                # calculate rms between the current solution and the altimetry data
        avg_solution_size = round(avg_solution_size / len(all_time_steps))
        avg_clustered_values = {}
        # If every station is its own cluster, then there are no differences
        if len(all_diffs) != 0:
            avg_diff_per_radius[radius] = numpy.mean(all_diffs)
            max_diff_per_radius[radius] = max_diff
        else:
            avg_diff_per_radius[radius] = None
            max_diff_per_radius[radius] = None
        for date in clustered_values.keys():
            if numpy.isnan(clustered_values[date]).any():
                print(f"Nan in clustered_values for date {date}: {numpy.isnan(clustered_values[date]).any()}")
            # check if data is empty
            if len(clustered_values[date]) == 0:
                print(f"Empty data for date {date}")
            avg_clustered_values[date] = numpy.mean(clustered_values[date])
        # mean center the values
        current_sum = 0
        for value in avg_clustered_values.values():
            current_sum += value
        mean = current_sum / len(avg_clustered_values)
        avg_clustered_values_mean_centered = {}
        for date in avg_clustered_values.keys():
            avg_clustered_values_mean_centered[date] = avg_clustered_values[date] - mean
        plot.plot_timelines([(avg_clustered_values_mean_centered, f"clustered stations radius {radius}", "green"),
                             (current_altimetry_data, "altimetry", "blue")],
                            f"comparison_radius_{radius}", output_directory)
        plot.plot_timelines([(timeline_all_stations, "all stations", "red"),
                             (avg_clustered_values_mean_centered, f"clustered stations radius {radius}", "green"),
                             (current_altimetry_data, "altimetry", "blue")],
                            f"clustered_global_mean_all_stations{radius}", output_directory)
        timeline = list(avg_clustered_values_mean_centered.keys())
        dates_mapping = find_closest_date(current_altimetry_data, timeline)
        to_delete = []
        for date in current_altimetry_data.keys():
            if date not in dates_mapping.values():
                to_delete.append(date)
        for date in to_delete:
            del current_altimetry_data[date]
        with open(os.path.join(output_directory, f"date_mapping.json"), "w") as file:
            json.dump(dates_mapping, file)
        rms = calculate_rms(current_altimetry_data, avg_clustered_values_mean_centered, dates_mapping)
        rms_per_radius[radius] = rms
        rms_per_number_of_clusters[avg_solution_size] = rms

    return rms_per_radius, rms_per_number_of_clusters


def calculate_rms_all_stations(stations: {int: timeseries_difference.TideGaugeStation}, all_radii: [float],
                               altimetry: {float: float}, output_directory: str):
    """
    Calculate the root-mean-square error for all stations
    :param stations:
    :param all_radii:
    :param altimetry:
    :param output_directory:
    :return:
    """
    rms_for_radius = {}
    mean_all_stations = {}
    collect_stations = {}
    for station_id in stations.keys():
        current_station = stations[station_id]
        current_timeseries = current_station.timeseries
        # calculate rms between the current solution and the altimetry data
        for date in current_timeseries.keys():
            if date >= 1992.9:
                if current_timeseries[date] == -99999:
                    continue
                if date not in collect_stations.keys():
                    collect_stations[date] = [current_timeseries[date]]
                else:
                    collect_stations[date].append(current_timeseries[date])
    for date in collect_stations.keys():
        if numpy.isnan(collect_stations[date]).any():
            print(f"Nan in collect_stations for date {date}: {numpy.isnan(collect_stations[date]).any()}")
        if len(collect_stations[date]) == 0:
            print(f"Empty data for date {date}")
        mean_all_stations[date] = numpy.mean(collect_stations[date])
    plot.plot_timelines([(mean_all_stations, "mean all stations", "blue")], "mean_all_stations", output_directory)
    # mean center the values
    current_sum = 0
    for value in mean_all_stations.values():
        current_sum += value
    mean = current_sum / len(mean_all_stations)
    for date in mean_all_stations.keys():
        mean_all_stations[date] = mean_all_stations[date] - mean

    timeline = list(mean_all_stations.keys())
    dates_mapping = find_closest_date(altimetry, timeline)
    dates_to_remove = []
    for date in altimetry.keys():
        if date not in dates_mapping.values():
            dates_to_remove.append(date)
    for date in dates_to_remove:
        del altimetry[date]

    plot.plot_timelines([(mean_all_stations, "mean_all_stations_mean_centered", "red")],
                        "mean_all_stations_mean_centered",
                        output_directory)
    plot.plot_timelines([(altimetry, "altimetry", "blue")], "altimetry", output_directory)
    plot.plot_timelines([(mean_all_stations, "all stations", "red"), (altimetry, "altimetry", "blue")], "comparison",
                        output_directory)
    with open(os.path.join(output_directory, "date_mapping_all.json"), "w") as file:
        json.dump(dates_mapping, file)
    with open(os.path.join(output_directory, "altimetry.json"), "w") as file:
        json.dump(altimetry, file)
    rms = calculate_rms(altimetry, mean_all_stations, dates_mapping)
    for radius in all_radii:
        rms_for_radius[radius] = rms
    return rms_for_radius, mean_all_stations


def plot_save_rms(rms_all_stations: {float: float}, rms_clustered_stations: {float: float}, output_directory: str,
                  color: str):
    """
    Plot and save the root-mean-square error calculated between global sea level (altimetry) and the mean sea level of
    all stations and the clustered stations
    :param color:
    :param rms_all_stations:
    :param rms_clustered_stations:
    :param output_directory:
    :return:
    """
    with open(os.path.join(output_directory, "rms.json"), "w") as file:
        json.dump(rms_clustered_stations, file)
    # plot
    plot.plot_rmse_graph([(rms_clustered_stations, "RMSE clustered stations", color),
                          (rms_all_stations, "RMSE all stations", "red")], "rms", output_directory,
                         "Number of cluster centers")


if __name__ == "__main__":
    # color scheme: teal, firebrick, goldenrod, purple
    output_path = "../output/evaluation_section_clustering/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    altimetry_directory = "../data/global_sea_level_altimetry/gmsl_2023rel2_seasons_retained.txt"
    PSMSL_path = "../data/rlr_monthly/2010.rlrdata"
    altimetry_data = read_altimetry_data(altimetry_directory)
    station_data = src.inner.tide_gauge_station.read_and_create_stations("../data/rlr_monthly/filelist.txt",
                                                                         os.path.join(output_path, "metadata.txt"))

    time_steps = ["1992_2002", "2002_2012", "2012_2022", "2022_2032"]
    rms_section_clustering_color = "purple"
    rms_voronoi_color = "mediumblue"
    rms_random_sample_color = "firebrick"
    rms_random_equal_distribution_color = "goldenrod"
    rms_original_clustering_color = "teal"
    rms_all_stations_color = "slategray"
    # radii = [0.0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    #          100]
    radii = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    clustering_path = "../output/Voronoi/"
    rms_all, timeline_all = calculate_rms_all_stations(station_data, radii, altimetry_data, output_path)
    rms_clustered, rms_per_number_of_centers = evaluate_clustering(radii, time_steps, altimetry_data, station_data,
                                                                   timeline_all, output_path,
                                                                   clustering_path)

    # rms all per number of centers
    rms_all_stations_per_number_of_centers = {}
    for number_of_centers in rms_per_number_of_centers.keys():
        rms_all_stations_per_number_of_centers[number_of_centers] = rms_all[50]

    second_clustering_path = "../output/section_clustering/"
    second_clustering_path_output = "../output/section_clustering/evaluation/"
    if not os.path.exists(second_clustering_path_output):
        os.makedirs(second_clustering_path_output)
    rms_second_clustering, rms_per_number_of_centers_second_clustering = evaluate_clustering(radii, time_steps,
                                                                                             altimetry_data,
                                                                                             station_data,
                                                                                             timeline_all,
                                                                                             second_clustering_path_output,
                                                                                             second_clustering_path)

    original_clustering_path = "../output/Comparing_global_avg"
    original_clustering_path_output = "../output/Comparing_global_avg/evaluation/"
    radii_for_original = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 80, 85, 90, 95, 100]
    if not os.path.exists(original_clustering_path_output):
        os.makedirs(original_clustering_path_output)
    rms_original_clustering, rms_per_number_of_centers_original_clustering = evaluate_clustering(
        radii_for_original, time_steps,
        altimetry_data,
        station_data,
        timeline_all,
        original_clustering_path_output,
        original_clustering_path)
    to_remove = []
    for number_of_centers in rms_per_number_of_centers_original_clustering.keys():
        if number_of_centers > max(rms_per_number_of_centers.keys()):
            to_remove.append(number_of_centers)
    for number_of_centers in to_remove:
        del rms_per_number_of_centers_original_clustering[number_of_centers]
    radii = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    # calculate the difference between the randomly selected stations and the global sea level for the time steps
    random_selection_path = "../output/random_sample_for_section_clustering/v2"
    random_output_path = "../output/Comparing_global_avg/evaluation/random_sample/"
    os.makedirs(random_output_path, exist_ok=True)
    rms_random_sample, rms_random_sample_per_number_of_centers = evaluate_clustering(radii, time_steps, altimetry_data,
                                                                                     station_data,
                                                                                     timeline_all, random_output_path,
                                                                                     random_selection_path)

    rms_random_sample_per_number_of_centers[max(list(rms_per_number_of_centers.keys()))] = rms_per_number_of_centers[
        max(list(rms_per_number_of_centers.keys()))]

    # calculate the difference between the random-equal distributed stations and the global sea level for the time steps
    random_equal_distribution_path = "../output/random_sample_for_section_clustering/v2/equal_per_hemisphere"
    random_equal_distribution_output_path = \
        "../output/Comparing_global_avg/evaluation/random_sample_equal_distribution/"
    os.makedirs(random_equal_distribution_output_path, exist_ok=True)
    rms_random_equal_distribution, rms_random_equal_distribution_per_number_of_centers = evaluate_clustering(radii,
                                                                                                             time_steps,
                                                                                                             altimetry_data,
                                                                                                             station_data,
                                                                                                             timeline_all,
                                                                                                             random_equal_distribution_output_path,
                                                                                                             random_equal_distribution_path)

    rms_random_equal_distribution_per_number_of_centers[max(list(rms_per_number_of_centers.keys()))] = \
        rms_per_number_of_centers[max(list(rms_per_number_of_centers.keys()))]

    # # get rid of all keys over 650
    # to_remove = []
    # for number_of_centers in rms_per_number_of_centers.keys():
    #     if number_of_centers > 650:
    #         to_remove.append(number_of_centers)
    # for number_of_centers in to_remove:
    #     del rms_per_number_of_centers[number_of_centers]
    # to_remove = []
    # for number_of_centers in rms_random_sample_per_number_of_centers.keys():
    #     if number_of_centers > 650:
    #         to_remove.append(number_of_centers)
    # for number_of_centers in to_remove:
    #     del rms_random_sample_per_number_of_centers[number_of_centers]
    # to_remove = []
    # for number_of_centers in rms_random_equal_distribution_per_number_of_centers.keys():
    #     if number_of_centers > 650:
    #         to_remove.append(number_of_centers)
    # for number_of_centers in to_remove:
    #     del rms_random_equal_distribution_per_number_of_centers[number_of_centers]
    # to_remove = []
    # for number_of_centers in rms_per_number_of_centers_second_clustering.keys():
    #     if number_of_centers > 650:
    #         to_remove.append(number_of_centers)
    # for number_of_centers in to_remove:
    #     del rms_per_number_of_centers_second_clustering[number_of_centers]
    # to_remove = []
    # for number_of_centers in rms_all_stations_per_number_of_centers.keys():
    #     if number_of_centers > 650:
    #         to_remove.append(number_of_centers)
    # for number_of_centers in to_remove:
    #     del rms_all_stations_per_number_of_centers[number_of_centers]

    # plot all three clusterings together
    plot.plot_rmse_graph([(rms_per_number_of_centers, "voronoi connected clustering", rms_voronoi_color),
                          (rms_per_number_of_centers_original_clustering, "connected clustering",
                           rms_original_clustering_color),
                          (rms_per_number_of_centers_second_clustering, "equitable connected clustering",
                           rms_section_clustering_color)],
                         "rms_all_different_clusterings", output_path, x_label="Number of centers")
    # plot section and original clustering
    plot.plot_rmse_graph(
        [(rms_per_number_of_centers_original_clustering, "connected clustering", rms_original_clustering_color),
         (rms_per_number_of_centers_second_clustering, "equitable connected clustering", rms_section_clustering_color)],
        "rms_section_and_original_clusterings", output_path, x_label="Number of centers")

    # plot voronoi and all stations
    plot.plot_rmse_graph([(rms_per_number_of_centers, "voronoi connected clustering", rms_voronoi_color),
                          (rms_all_stations_per_number_of_centers, "all stations", rms_all_stations_color)],
                         "rmse_clustered_all",
                         output_path, x_label="Number of centers")
    # plot random sample, voronoi clustering and all stations
    plot.plot_rmse_graph([(rms_per_number_of_centers, "voronoi connected clustering", rms_voronoi_color),
                          (rms_random_sample_per_number_of_centers, "random stations", rms_random_sample_color),
                          (rms_all_stations_per_number_of_centers, "all stations", rms_all_stations_color)],
                         "rms_voronoi_random", output_path, x_label="Number of centers")
    # plot voronoi, random equal, all stations
    plot.plot_rmse_graph([(rms_per_number_of_centers, "voronoi connected clustering", rms_voronoi_color),
                          (rms_random_equal_distribution_per_number_of_centers, "random equal distribution",
                           rms_random_equal_distribution_color),
                          (rms_all_stations_per_number_of_centers, "all stations", rms_all_stations_color)],
                         "rms_voronoi_clustered_random_equal",
                         output_path, x_label="Number of centers")
    # plot random, random_equal, original, section clustering
    plot.plot_rmse_graph([(rms_all_stations_per_number_of_centers, "all stations", rms_all_stations_color),
                          (rms_random_sample_per_number_of_centers, "random stations", rms_random_sample_color),
                          (rms_random_equal_distribution_per_number_of_centers, "random equal distribution",
                           rms_random_equal_distribution_color),
                          (rms_per_number_of_centers_original_clustering, "connected clustering",
                           rms_original_clustering_color),
                          (rms_per_number_of_centers_second_clustering, "equitable connected clustering",
                           rms_section_clustering_color)],
                         "rms_all_different_clusterings_for_section_clustering", output_path,
                         x_label="Number of centers")
