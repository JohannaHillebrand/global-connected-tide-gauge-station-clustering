import glob
import json
import math
import os

import cartopy.crs as ccrs
import geopandas
import netCDF4
import numpy
import numpy as np
import regionmask
import scipy
import seaborn
import shapely
import xarray as xr
from astropy.time import Time
from loguru import logger
from matplotlib import pyplot as plt

import src.inner.tide_gauge_station
from src.inner import timeseries_difference
from src.inner.plot import plot_timelines, plot_rmse_graph

global GLOBAL_MEAN_SEA_LEVEL
global OUTPUT_DIRECTORY
global STATIONS
global MEAN_SEA_LEVEL_ALL_STATIONS
global INTERVALS
global METADATA_FILE_PATH
global RADII
global CLUSTERING_INPUT_PATH


def init(oras5_path, oras_dim_path, output, stations, intervals, all_radii, clustering_input_path):
    global GLOBAL_MEAN_SEA_LEVEL
    global OUTPUT_DIRECTORY
    global STATIONS
    global MEAN_SEA_LEVEL_ALL_STATIONS
    global INTERVALS
    global METADATA_FILE_PATH
    global RADII
    global CLUSTERING_INPUT_PATH
    GLOBAL_MEAN_SEA_LEVEL = calculate_global_mean_sea_level(oras5_path, oras_dim_path)
    OUTPUT_DIRECTORY = output
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    STATIONS = src.inner.tide_gauge_station.read_and_create_stations(stations,
                                                                     os.path.join(OUTPUT_DIRECTORY, "metadata.txt"))
    STATIONS = src.inner.tide_gauge_station.detrend_and_mean_center_timeseries(STATIONS)
    MEAN_SEA_LEVEL_ALL_STATIONS = calculate_mean_sea_level_all_tide_gauge_stations(STATIONS)
    INTERVALS = intervals
    METADATA_FILE_PATH = os.path.join(OUTPUT_DIRECTORY, "metadata.txt")
    RADII = all_radii
    CLUSTERING_INPUT_PATH = clustering_input_path
    logger.info("Initialization complete")


def calculate_global_mean_sea_level(oras5_path: str, oras5_dim_path: str):
    """
    Calculate the global mean sea level based on the ORAS5 data
    :return:
    """
    counter = 0
    # read grid, which gives us the dimensions of each grid point
    oras5_dim = netCDF4.Dataset(oras5_dim_path)
    e1t = oras5_dim.variables['e1t'][:]
    e2t = oras5_dim.variables['e2t'][:]
    # This is an array, which contains the area of each grid point
    grid_dimensions = numpy.multiply(e1t, e2t)
    average_global_sea_level = {}
    directories = [x[0] for x in os.walk(oras5_path)][1:]
    for directory in directories:
        files = glob.glob(f"{directory}/*.nc")
        for file in files:
            # counter += 1
            # if counter == 2:
            #     exit(0)
            oras5 = netCDF4.Dataset(file)
            current_date_netcdf = netCDF4.num2date(oras5.variables["time_counter"][:],
                                                   oras5.variables["time_counter"].units,
                                                   only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            astropy_time_object = Time(current_date_netcdf[0], format="datetime", scale="utc")
            current_date = astropy_time_object.decimalyear
            lat = oras5.variables["nav_lat"][:]
            lon = oras5.variables["nav_lon"][:]
            time = oras5.variables["time_counter"][:]
            sossheig = oras5.variables["sossheig"][:]
            sossheig = sossheig.squeeze()
            # sossheig = replace_mask_with_None(sossheig)
            # plot_world_map(lat, lon, sossheig, grid_dimensions)
            # weighted mean: sum(w * sossheig) / sum(w)
            multiplied = numpy.multiply(grid_dimensions, sossheig)
            weighted_mean_sea_level = numpy.sum(multiplied) / numpy.sum(grid_dimensions)
            average_global_sea_level[current_date] = weighted_mean_sea_level
            oras5.close()
    oras5_dim.close()
    return average_global_sea_level


def getclosest_ij(lats, lons, latpt, lonpt):
    """
    Find the closest point in the grid to the given point
    :param lats:
    :param lons:
    :param latpt:
    :param lonpt:
    :return:
    """
    # find squared distance of every point on grid
    dist_sq = (lats - latpt) ** 2 + (lons - lonpt) ** 2
    # 1D index of minimum dist_sq element
    minindex_flattened = dist_sq.argmin()
    # Get 2D index for latvals and lonvals arrays from 1D index
    return np.unravel_index(minindex_flattened, lats.shape)


def replace_mask_with_None(sossheig):
    zero_counter = 0
    non_zero_counter = 0
    for i in sossheig.recordmask:
        for j in i:
            # these are masked arrays, find missing data
            if j == True:
                zero_counter += 1
            if j == False:
                non_zero_counter += 1
    print(f" zero values {zero_counter}")
    print(f" non zero values {non_zero_counter}")


def plot_world_map(lat, lon, sossheig, w):
    """
    Plot the world map
    :param getclosest_ij:
    :param lat:
    :param lon:
    :param sossheig:
    :param w:
    :return:
    """
    sossheig = numpy.multiply(sossheig, w)
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=1)
    ax.gridlines(linestyle="--", color="black")
    plt.contourf(lon, lat, sossheig, 60, transform=ccrs.PlateCarree(), cmap="viridis")
    plt.colorbar(label="sea level")
    plt.title("sea level")
    plt.savefig("../output/sealevelgrid.png")
    plt.close()
    return sossheig


def calculate_RMS(current_clustered_solution: {float, float}, global_mean_sea_level):
    """
    Calculate the root mean square error between the global mean sea level and the clustered solution
    :param current_clustered_solution: 
    :param global_mean_sea_level: 
    :return: 
    """""
    sorted_current_clustered_solution = dict(sorted(current_clustered_solution.items()))
    sorted_global_mean_sea_level = dict(sorted(global_mean_sea_level.items()))
    # calculate the root mean square error
    sum = 0
    for date in sorted_current_clustered_solution.keys():
        sum += (sorted_current_clustered_solution[date] - sorted_global_mean_sea_level[date]) ** 2
    rms = numpy.sqrt(sum / len(sorted_current_clustered_solution))
    return rms


def evaluate_clustering_for_time_steps(mean_sea_level_all_stations_mean_centered: {float, float}):
    """
    Evaluate the clustering per time step
    :param mean_sea_level_all_stations_mean_centered:
    :return:
    """
    all_radii_rmse = {}
    all_radii_rmse_mean_centered = {}
    avg_number_of_clusters_per_radius = {}

    for current_radius in RADII:
        # first plot for a specific date which stations are over and under the global mean sea level
        # plot_stations_over_under_global_sea_level(current_radius)
        # evaluate the clustering for each time step
        (rmse, rmse_mean_centered, avg_number_of_clusters) = evaluate_clustering_per_radius(
            current_radius,
            mean_sea_level_all_stations_mean_centered)
        all_radii_rmse[current_radius] = rmse
        all_radii_rmse_mean_centered[current_radius] = rmse_mean_centered
        avg_number_of_clusters_per_radius[current_radius] = avg_number_of_clusters

    return all_radii_rmse, all_radii_rmse_mean_centered, avg_number_of_clusters_per_radius


def evaluate_clustering_per_radius(current_radius, mean_sea_level_all_stations_normalized):
    """
    Evaluate the clustering per radius
    :param current_radius:
    :param mean_sea_level_all_stations_normalized:
    :return:
    """
    avg_number_of_clusters = 0
    complete_clustered_solution = {}
    complete_clustered_solution_mean_centered = {}
    list_number_of_clusters = []
    max_number_of_clusters = 0
    min_number_of_clusters = math.inf
    for timestep in INTERVALS:
        start_year = timestep[0]
        end_year = timestep[1]
        clustered_solution, number_of_clusters = clustered_mean_sea_level_per_time_step(
            start_year, end_year, current_radius)
        avg_number_of_clusters += number_of_clusters
        list_number_of_clusters.append(number_of_clusters)
        if number_of_clusters > max_number_of_clusters:
            max_number_of_clusters = number_of_clusters
        if number_of_clusters < min_number_of_clusters:
            min_number_of_clusters = number_of_clusters
        for date in clustered_solution.keys():
            complete_clustered_solution[date] = clustered_solution[date]
    avg_number_of_clusters = avg_number_of_clusters / len(INTERVALS)
    median_number_of_clusters = numpy.median(list_number_of_clusters)
    # normalize the clustered solution over the entire timespan
    avg_clustered_solution = sum(complete_clustered_solution.values()) / len(complete_clustered_solution)
    for date in complete_clustered_solution.keys():
        complete_clustered_solution_mean_centered[date] = complete_clustered_solution[date] - avg_clustered_solution

    plot_timelines([(GLOBAL_MEAN_SEA_LEVEL, "global sea level", "blue"),
                    (MEAN_SEA_LEVEL_ALL_STATIONS, "sea level all stations", "red"),
                    (complete_clustered_solution, "clustered solution", "green")], f"{current_radius}",
                   OUTPUT_DIRECTORY)
    plot_timelines([(GLOBAL_MEAN_SEA_LEVEL, "global sea level", "blue"),
                    (mean_sea_level_all_stations_normalized,
                     "sea level all stations, mean centered timeseries",
                     "red"),
                    (complete_clustered_solution_mean_centered, "clustered solution, mean centered timeseries",
                     "green")],
                   f"{current_radius}_mean_centered", OUTPUT_DIRECTORY)

    # calculate RMSE
    rms_clustering_global_average = calculate_RMS(complete_clustered_solution, GLOBAL_MEAN_SEA_LEVEL)
    rms_mean_centered_clustering_global_average = calculate_RMS(complete_clustered_solution_mean_centered,
                                                                GLOBAL_MEAN_SEA_LEVEL)
    # save RMSE for this radius
    write_rms_to_file(avg_number_of_clusters, current_radius, rms_clustering_global_average,
                      rms_mean_centered_clustering_global_average, median_number_of_clusters, max_number_of_clusters,
                      min_number_of_clusters)
    return (rms_clustering_global_average, rms_mean_centered_clustering_global_average, avg_number_of_clusters)


def write_rms_to_file(avg_number_of_clusters: float, current_radius: float, rms_clustering_global_average: float,
                      rms_normalized_clustering_global_average: float, median_number_of_clusters: float,
                      max_number_of_clusters: float, min_number_of_clusters: float):
    """
    Write the RMSE to the metadata file
    :param min_number_of_clusters:
    :param max_number_of_clusters:
    :param median_number_of_clusters:
    :param avg_number_of_clusters:
    :param current_radius:
    :param rms_clustering_global_average:
    :param rms_normalized_clustering_global_average:
    :return:
    """
    with open(os.path.join(OUTPUT_DIRECTORY, "metadata.txt"), "a") as file:
        file.write(
            f"RMSE between global mean sea level and clustered solution for radius {current_radius}: "
            f"{rms_clustering_global_average}\n")
        file.write(
            f"RMSE between global mean sea level and clustered solution MEAN CENTERED for radius {current_radius}: "
            f"{rms_normalized_clustering_global_average}\n")
        file.write(
            f"average number of clusters for radius {current_radius}: "
            f"{avg_number_of_clusters}\n")
        file.write(
            f"median number of clusters for radius {current_radius}: "
            f"{median_number_of_clusters}\n")
        file.write(
            f"max number of clusters for radius {current_radius}: "
            f"{max_number_of_clusters}\n")
        file.write(
            f"min number of clusters for radius {current_radius}: "
            f"{min_number_of_clusters}\n\n")
        file.write(f"---------------------------------------------\n\n")


def clustered_mean_sea_level_per_time_step(start_year: int, end_year: int, current_radius):
    """
    Calculate the mean clustered sea level per time step
    :param end_year:
    :param start_year:
    :param current_radius:
    :return:
    """
    clustered_solution = {}

    file_path = (f"{CLUSTERING_INPUT_PATH}/{start_year}_{end_year}/solution_"
                 f"{current_radius}.json")
    all_current_stations = {}
    with open(file_path) as file:
        current_solution = json.load(file)
    number_of_clusters = len(current_solution)
    for station_id in current_solution.keys():
        current_station = STATIONS[int(station_id)]
        all_current_stations[station_id] = current_station
        for date in current_station.timeseries.keys():
            if date >= start_year and date <= end_year:
                if date not in clustered_solution.keys():
                    clustered_solution[date] = current_station.timeseries[date]
                else:
                    clustered_solution[date] += current_station.timeseries[date]

    for date in clustered_solution.keys():
        if date >= start_year and date <= end_year:
            clustered_solution[date] = clustered_solution[date] / len(current_solution)

    return clustered_solution, number_of_clusters


def save_clustered_solution_all_time_steps(average_global_sea_level, normalized_global_sea_level,
                                           complete_clustered_solution,
                                           complete_clustered_solution_normalized, current_radius,
                                           mean_sea_level_all_stations, mean_sea_level_all_stations_normalized, out_dir,
                                           stations):
    """
    Save the clustered solution for all time steps
    :param normalized_global_sea_level:
    :param average_global_sea_level:
    :param complete_clustered_solution:
    :param complete_clustered_solution_normalized:
    :param current_radius:
    :param mean_sea_level_all_stations:
    :param mean_sea_level_all_stations_normalized:
    :param out_dir:
    :param stations:
    :return:
    """
    # calculate RMSE
    rms_clustering_global_average = calculate_RMS(complete_clustered_solution, average_global_sea_level)
    rms_normalized_clustering_global_average = calculate_RMS(complete_clustered_solution_normalized,
                                                             average_global_sea_level)
    rms_clustering_normalized_global_average = calculate_RMS(complete_clustered_solution,
                                                             normalized_global_sea_level)
    rms_normalized_clustering_normalized_global_average = calculate_RMS(complete_clustered_solution_normalized,
                                                                        normalized_global_sea_level)
    return (rms_clustering_global_average, rms_normalized_clustering_global_average,
            rms_clustering_normalized_global_average, rms_normalized_clustering_normalized_global_average)


def evaluate_one_clustering_solution(clustering_path, stations, average_global_sea_level,
                                     mean_sea_level_all_stations, mean_sea_level_all_stations_normalized,
                                     metadata_file_path: str, output_path: str):
    """
    Evaluate one clustering solution
    :param output_path:
    :param mean_sea_level_all_stations_normalized:
    :param metadata_file_path:
    :param clustering_path:
    :param stations:
    :param average_global_sea_level:
    :param mean_sea_level_all_stations:
    :return:
    """
    RMSE = {}
    RMSE_normalized = {}
    radii = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    for radius in radii:
        clustered_solution = {}
        clustered_solution_normalized = {}
        file_path = os.path.join(clustering_path, f"solution_{radius}.json")
        with open(file_path) as file:
            current_solution = json.load(file)
        for station_id in current_solution.keys():
            current_station = stations[int(station_id)]
            for date in current_station.timeseries.keys():
                if date not in clustered_solution.keys():
                    clustered_solution[date] = current_station.timeseries[date]
                    clustered_solution_normalized[date] = current_station.timeseries_detrended_normalized[date]
                else:
                    clustered_solution[date] += current_station.timeseries[date]
                    clustered_solution_normalized[date] += current_station.timeseries_detrended_normalized[date]
        for date in clustered_solution.keys():
            clustered_solution[date] = clustered_solution[date] / len(current_solution)
            clustered_solution_normalized[date] = clustered_solution_normalized[date] / len(current_solution)
        # plot
        fig, ax = plt.subplots()
        ax.plot(*zip(*sorted(average_global_sea_level.items())), color="blue", zorder=1,
                label="mean global sea level")
        ax.plot(*zip(*sorted(mean_sea_level_all_stations.items())), color="red", zorder=2,
                label="all tide gauge stations")
        ax.plot(*zip(*sorted(clustered_solution.items())), color="green", zorder=3,
                label=f"clustered stations {radius}m")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("sea level")
        plt.savefig(f"{output_path}all_{radius}.png")
        plt.close()
        fig, ax = plt.subplots()
        ax.plot(*zip(*sorted(average_global_sea_level.items())), color="blue", zorder=1,
                label="mean global sea level")
        ax.plot(*zip(*sorted(mean_sea_level_all_stations_normalized.items())), color="red", zorder=2,
                label="all tide gauge stations")
        ax.plot(*zip(*sorted(clustered_solution_normalized.items())), color="green", zorder=3,
                label=f"clustered stations {radius}m normalized")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("sea level")
        plt.savefig(f"{output_path}/normalized/all_{radius}.png")
        plt.close()
        rms_clustering_global_average = calculate_RMS(clustered_solution, average_global_sea_level)
        rms_clustering_global_average_normalized = calculate_RMS(clustered_solution_normalized,
                                                                 average_global_sea_level)
        RMSE[radius] = rms_clustering_global_average
        RMSE_normalized[radius] = rms_clustering_global_average_normalized
        with open(metadata_file_path, "a") as file:
            file.write(
                f"RMSE between global mean sea level and clustered solution for radius {radius}: "
                f"{rms_clustering_global_average}\n")
            file.write(
                f"RMSE between global mean sea level and clustered solution NORMALIZED for radius {radius}: "
                f"{rms_clustering_global_average_normalized}\n")
    return RMSE, RMSE_normalized


def write_rmse_to_file(rmse_clustering, rmse_clustering_mean_centered, rmse_all_stations,
                       rmse_all_stations_mean_centered):
    """
    Write the RMSE to the metadata file
    :param rmse_clustering:
    :param rmse_clustering_mean_centered:
    :param rmse_all_stations:
    :param rmse_all_stations_mean_centered:
    :return:
    """
    with open(METADATA_FILE_PATH, "a") as file:
        file.write("\n")
        file.write("RMSE\n")
        file.write("\n")
        file.write(
            f"RMSE between global mean sea level and mean sea level of all tide gauge stations: "
            f"{rmse_all_stations}\n\n")

        for radius in rmse_clustering.keys():
            file.write(f"RMSE between global mean sea level and clustered solution for radius {radius}: "
                       f"{rmse_clustering[radius]}\n\n")
        file.write(
            f"RMSE between global mean sea level and mean sea level of all tide gauge stations mean centered: "
            f"{rmse_all_stations_mean_centered}\n\n")
        for radius in rmse_clustering_mean_centered.keys():
            file.write(f"RMSE between global mean sea level and clustered solution MEAN CENTERED for radius {radius}: "
                       f"{rmse_clustering_mean_centered[radius]}\n")


def calculate_mean_sea_level_all_tide_gauge_stations(stations):
    """
    Calculate the mean sea level of all tide gauge stations
    :param stations:
    :return:
    """
    sea_level_sum = {}
    for station_id in stations.keys():
        current_station = stations[station_id]
        for date in current_station.timeseries.keys():
            if date not in sea_level_sum.keys():
                sea_level_sum[date] = current_station.timeseries[date]
            else:
                sea_level_sum[date] += current_station.timeseries[date]
    average_global_sea_level_all_tide_gauge_stations = {}
    for date in sea_level_sum.keys():
        average_global_sea_level_all_tide_gauge_stations[date] = sea_level_sum[date] / len(stations)

    return average_global_sea_level_all_tide_gauge_stations


def normalize_detrend_avg_global_sea_level():
    """
    Normalize and detrend the average global sea level for a better comparison (for each 10-year interval)
    :return:
    """
    normalized_avg_global_sea_level = {}
    for timestep in INTERVALS:
        start_year = timestep[0]
        end_year = timestep[1]
        current_avg_global_sea_level = {}
        for date in GLOBAL_MEAN_SEA_LEVEL.keys():
            if date >= start_year and date < end_year:
                current_avg_global_sea_level[date] = GLOBAL_MEAN_SEA_LEVEL[date]
        # detrend & normalize the current 10 year interval
        dates = []
        sea_level = []
        sorted_timeseries = dict(sorted(current_avg_global_sea_level.items()))
        for date in sorted_timeseries.keys():
            dates.append(date)
            sea_level.append(sorted_timeseries[date])
        if len(dates) > 1:
            detrended_sea_level = scipy.signal.detrend(sea_level)
            sum_detrended_sea_level = sum(detrended_sea_level)
            avg_detrended_sea_level = sum_detrended_sea_level / len(detrended_sea_level)
            normalized_and_detrended_sea_level = [x - avg_detrended_sea_level for x in detrended_sea_level]
            for i in range(len(dates)):
                current_avg_global_sea_level[dates[i]] = normalized_and_detrended_sea_level[i]
        for date in current_avg_global_sea_level.keys():
            normalized_avg_global_sea_level[date] = current_avg_global_sea_level[date]
    # plot
    fig, ax = plt.subplots()
    ax.plot(*zip(*sorted(normalized_avg_global_sea_level.items())), color="blue", zorder=1,
            label="normalized, detrended global sea level")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("sea level")
    plt.savefig(f"{OUTPUT_DIRECTORY}normalized_avg_global_sea_level.png")
    plt.close()

    return normalized_avg_global_sea_level


def detrend_and_normalize_each_10_year_interval(stations: {int, timeseries_difference.TideGaugeStation},
                                                intervals: [(int, int)], metadata_path: str):
    """
    Detrend and normalize each 10-year interval for all stations
    :param metadata_path:
    :param intervals:
    :param stations:
    :return:
    """
    for station_id in stations.keys():
        stations[station_id].timeseries_detrended_normalized = {}
        stations = stations.copy()
    for time_step in intervals:
        start_year = time_step[0]
        end_year = time_step[1]
        current_stations = timeseries_difference.remove_dates_before_and_after_threshold(start_year, end_year, stations,
                                                                                         metadata_path)
        current_stations = src.inner.tide_gauge_station.detrend_and_mean_center_timeseries(current_stations)
        for station_id in current_stations.keys():
            for date in current_stations[station_id].timeseries.keys():
                stations[station_id].timeseries_detrended_normalized[date] = \
                    current_stations[station_id].timeseries_detrended_normalized[
                        date]

    return stations


def mean_center_sea_level():
    """
    Normalize the mean sea level, as we are interested in the relative sea level behaviour and not the absolute sea
    level behaviour
    :return:
    """
    mean_sea_level_all_stations_mean_centered = {}
    avg_mean_sea_level = sum(MEAN_SEA_LEVEL_ALL_STATIONS.values()) / len(MEAN_SEA_LEVEL_ALL_STATIONS)
    for date in MEAN_SEA_LEVEL_ALL_STATIONS.keys():
        mean_sea_level_all_stations_mean_centered[date] = MEAN_SEA_LEVEL_ALL_STATIONS[date] - avg_mean_sea_level
    return mean_sea_level_all_stations_mean_centered


def start(time_steps: bool):
    """
    Start the calculation of the global mean sea level
    :param time_steps:
    :return:
    """

    normalized_avg_global_sea_level = normalize_detrend_avg_global_sea_level()
    plot_timelines([(GLOBAL_MEAN_SEA_LEVEL, "global sea level", "blue"),
                    (normalized_avg_global_sea_level, "global sea level normalized, detrended", "lightblue")],
                   "global_sea_level", OUTPUT_DIRECTORY)
    # calculate the normalized mean sea level of all tide gauge stations
    mean_sea_level_all_stations_mean_centered = mean_center_sea_level()
    plot_timelines([
        (MEAN_SEA_LEVEL_ALL_STATIONS, "sea level based on all stations", "green"),
        (mean_sea_level_all_stations_mean_centered, "sea level based on all stations mean centered", "red"),
        (GLOBAL_MEAN_SEA_LEVEL, "global sea level", "blue")],
        "sea_level_all_stations", OUTPUT_DIRECTORY)
    # normalize & detrend each 10 year intervall for all stations and calculate the mean sea level based on this
    # stations_normalized_detrended = detrend_and_normalize_each_10_year_interval(stations, intervals, meta_file_path)
    # mean_sea_level_all_stations, mean_sea_level_all_stations_normalized_detrended = (
    #     calculate_mean_sea_level_all_tide_gauge_stations(stations_normalized_detrended))
    # plot_timelines([(mean_sea_level_all_stations_normalized_detrended,
    #                  "sea level based on all stations normalized, detrended",
    #                  "red"), (mean_sea_level_all_stations, "sea level based on all stations", "green"),
    #                 (average_global_sea_level, "global sea level", "blue"),
    #                 (normalized_avg_global_sea_level, "global sea level normalized, detrended", "lightblue")],
    #                output_directory, name="sea_level_all_stations_normalized_detrended")

    # add clustered stations
    if time_steps:
        (RMSE, RMSE_mean_centered, avg_number_of_clusters_per_radius) = evaluate_clustering_for_time_steps(
            mean_sea_level_all_stations_mean_centered)
    else:
        pass
        # RMSE, RMSE_mean_centered, avg_number_of_clusters_per_radius = evaluate_one_clustering_solution(
        #     mean_sea_level_all_stations_mean_centered)

    # calculate RMSE between global mean sea level and mean sea level of all tide gauge stations
    rms_all_stations_global_average = calculate_RMS(MEAN_SEA_LEVEL_ALL_STATIONS, GLOBAL_MEAN_SEA_LEVEL)
    rms_all_stations_global_average_normalized = calculate_RMS(mean_sea_level_all_stations_mean_centered,
                                                               GLOBAL_MEAN_SEA_LEVEL)

    write_rmse_to_file(RMSE, RMSE_mean_centered, rms_all_stations_global_average,
                       rms_all_stations_global_average_normalized)

    rmse_all_stations = {}
    rmse_mean_centered_all_stations = {}
    # The RMSE values do not change for the all stations solution, this is just for plotting purposes
    for radius in RMSE.keys():
        rmse_all_stations[radius] = rms_all_stations_global_average
        rmse_mean_centered_all_stations[radius] = rms_all_stations_global_average_normalized
    plot_rmse_graph(
        [(RMSE, "RMS clustering", "lightblue"), (RMSE_mean_centered, "RMS clustering mean centered", "lightgreen"),
         (rmse_all_stations, "RMS all stations", "blue"),
         (rmse_mean_centered_all_stations, "RMS all stations, mean centered", "green")],
        "rmse_clustering_mean_centered", OUTPUT_DIRECTORY)


def plot_stations_over_under_global_sea_level(radius: float):
    """
    Plot the stations that are over or under the global sea level
    :param radius:
    :return:
    """
    if not os.path.exists(f"{OUTPUT_DIRECTORY}/over_under"):
        os.makedirs(f"{OUTPUT_DIRECTORY}/over_under")
    # read in stations that are center points in the current clustering solution and plot them
    time_step = "2014_2024"
    file_path = (f"{CLUSTERING_INPUT_PATH}/{time_step}/solution_"
                 f"{radius}.json")
    all_current_stations = {}
    with open(file_path) as file:
        current_solution = json.load(file)

    # plot all stations for date: 2020.042349726776, if they are over the global sea level make them red,
    # otherwise blue
    stations_over = {"id": [], "color": [], "geometry": []}
    how_much_over = []
    stations_under = {"id": [], "color": [], "geometry": []}
    how_much_under = []
    for station_id in current_solution.keys():
        station = STATIONS[int(station_id)]
        if station.timeseries[2020.042349726776] > GLOBAL_MEAN_SEA_LEVEL[2020.042349726776]:
            stations_over["id"].append(station.id)
            stations_over["color"].append("red")
            stations_over["geometry"].append(shapely.Point(station.longitude, station.latitude))
            how_much_over.append(station.timeseries[2020.042349726776] - GLOBAL_MEAN_SEA_LEVEL[2020.042349726776])
        else:
            stations_under["id"].append(station.id)
            stations_under["color"].append("blue")
            stations_under["geometry"].append(shapely.Point(station.longitude, station.latitude))
            how_much_under.append(station.timeseries[2020.042349726776] - GLOBAL_MEAN_SEA_LEVEL[2020.042349726776])
    # make heatmap
    all_values = sorted(how_much_under + how_much_over, reverse=True)
    side_length = math.ceil(math.sqrt(len(all_values)))
    zeroes = numpy.empty(side_length * side_length)

    for i in range(len(all_values)):
        zeroes[i] = all_values[i]
    heatmap_array = numpy.array((zeroes))
    heatmap_array_shaped = heatmap_array.reshape(side_length, side_length)
    fig, ax = plt.subplots()
    seaborn.heatmap(heatmap_array_shaped, ax=ax, vmin=min(all_values), vmax=max(all_values), center=0, cmap="bwr")
    seaborn.color_palette("colorblind")
    plt.savefig(f"{OUTPUT_DIRECTORY}/over_under/{radius}heatmap.png")
    plt.close()
    # write to file
    with open(f"{OUTPUT_DIRECTORY}/over_under/stations_over_under_global_sea_level.txt", "a") as file:
        file.write(f"radius: {radius}\n")
        file.write(f"stations over global sea level: {len(stations_over['id'])}\n")
        file.write(f"stations under global sea level: {len(stations_under['id'])}\n")
    # make gdf from dict
    gdf_over = geopandas.GeoDataFrame(stations_over)
    gdf_over.set_geometry("geometry")
    gdf_under = geopandas.GeoDataFrame(stations_under)
    gdf_under.set_geometry("geometry")
    # plot
    land_path = "../../data/ne_10m_land/ne_10m_land.shp"
    fig, ax = plt.subplots()
    land_gdf = geopandas.read_file(land_path)
    land_gdf = land_gdf.explode("geometry", ignore_index=True)
    ax = land_gdf.plot(color="burlywood", zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    gdf_over.plot(ax=ax, color="red", marker=",", markersize=1, zorder=1)
    gdf_under.plot(ax=ax, color="blue", marker=",", markersize=1, zorder=1)
    plt.savefig(f"{OUTPUT_DIRECTORY}/over_under/{radius}stations_over_under_global_sea_level.png", dpi=500)
    plt.close("all")


def evaluate_selected_regions(oras5_path: str, oras5_dim_path: str):
    """
    check the similarity between the clustered stations in the mediterranean sea and the grid data for that region
    :return:
    """
    xr.set_options(keep_attrs=True)
    logger.info(f"Mask creation")
    oras_dim = xr.open_dataset(oras5_dim_path)
    e1t = oras_dim.e1t[:]
    e2t = oras_dim.e2t[:]
    # This is an array, which contains the area of each grid point
    grid_dimensions = numpy.multiply(e1t, e2t)
    first_oras5 = xr.open_dataset(
        os.path.join(oras5_path, "1958-1979/sossheig_control_monthly_highres_2D_195801_CONS_v0.1.nc"))
    lon = first_oras5.nav_lon[:]
    lat = first_oras5.nav_lat[:]
    ocean_mask = regionmask.defined_regions.ar6.all.mask_3D(lon, lat)
    # print(ocean_mask.region.names)
    mediterranean_mask = ocean_mask.sel(region=19)
    second_mask = ocean_mask.sel(region=9)
    third_mask = ocean_mask.sel(region=13)
    fourth_mask = ocean_mask.sel(region=14)
    fifth_mask = ocean_mask.sel(region=15)

    mediterranean_values = first_oras5.where(mediterranean_mask)
    second_values = first_oras5.where(second_mask)
    third_values = first_oras5.where(third_mask)
    fourth_values = first_oras5.where(fourth_mask)
    fifth_values = first_oras5.where(fifth_mask)

    # sum up all values in the mediterranean sea and divide by the number of grid points
    sossheig_med = mediterranean_values.sossheig[0, :, :]
    sossheig_second = second_values.sossheig[0, :, :]
    sossheig_third = third_values.sossheig[0, :, :]
    sossheig_fourth = fourth_values.sossheig[0, :, :]
    sossheig_fifth = fifth_values.sossheig[0, :, :]

    # print(sossheig)
    logger.info(f"grid point selection")
    grid_point_counter = 0
    selected_coords = []

    for i in range(len(sossheig_med)):
        for j in range(len(sossheig_med[i])):
            if not numpy.isnan(sossheig_med[i][j]) or not numpy.isnan(sossheig_second[i][j]) or not numpy.isnan(
                    sossheig_third[i][j]) or not numpy.isnan(sossheig_fourth[i][j]) or not numpy.isnan(
                sossheig_fifth[i][j]):
                grid_point_counter += 1
                selected_coords.append((lon[i][j], lat[i][j]))

    # for i in range(len(sossheig_med)):
    #     for j in range(len(sossheig_med[i])):
    #         if not numpy.isnan(sossheig_med[i][j]):
    #             grid_point_counter += 1
    #             selected_coords.append((lon[i][j], lat[i][j]))
    # logger.info(f"number of grid points mediterranean: {grid_point_counter}")
    # for i in range(len(sossheig_second)):
    #     for j in range(len(sossheig_second[i])):
    #         if not numpy.isnan(sossheig_second[i][j]):
    #             grid_point_counter += 1
    #             selected_coords.append((lon[i][j], lat[i][j]))
    logger.info(f"number of grid points overall: {grid_point_counter}")
    # plot grid point
    logger.info(f"plotting grid points")
    land_gdf = geopandas.read_file(land_directory)
    land_gdf = land_gdf.explode("geometry", ignore_index=True)
    grid_dict = {"id": [], "geometry": []}
    counter = 0
    for coords in selected_coords:
        counter += 1
        grid_dict["id"].append(counter)
        grid_dict["geometry"].append(shapely.geometry.Point(coords[0], coords[1]))
    grid_gdf = geopandas.GeoDataFrame(grid_dict)
    ax = land_gdf.plot(color="burlywood", figsize=(40, 24), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    grid_gdf.plot(ax=ax, color="red", marker=",", markersize=0.01, zorder=1)
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, "grid_points.pdf"))
    plt.close()

    average_region_sea_level = {}
    directories = [x[0] for x in os.walk(oras5_path)][1:]
    for directory in directories:
        files = glob.glob(f"{directory}/*.nc")
        for file in files:
            # counter += 1
            # if counter == 2:
            #     exit(0)
            oras5_netdcf = netCDF4.Dataset(file)
            oras5_xr = xr.open_dataset(file)
            current_date_netcdf = netCDF4.num2date(oras5_netdcf.variables["time_counter"][:],
                                                   oras5_netdcf.variables["time_counter"].units,
                                                   only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            astropy_time_object = Time(current_date_netcdf[0], format="datetime", scale="utc")
            current_date = astropy_time_object.decimalyear
            values1 = oras5_xr.where(mediterranean_mask).sossheig[0, :, :]
            values2 = oras5_xr.where(second_mask).sossheig[0, :, :]
            values3 = oras5_xr.where(third_mask).sossheig[0, :, :]
            values4 = oras5_xr.where(fourth_mask).sossheig[0, :, :]
            values5 = oras5_xr.where(fifth_mask).sossheig[0, :, :]

            sum_values = (
                    values1.squeeze().sum() + values2.squeeze().sum() + values3.squeeze().sum() + values4.squeeze(
            ).sum() + values5.squeeze().sum())

            sea_level = sum_values.to_numpy() / grid_point_counter
            average_region_sea_level[current_date] = sea_level
    avg = 0
    for date in average_region_sea_level.keys():
        avg += average_region_sea_level[date]
    avg = avg / len(average_region_sea_level)
    for date in average_region_sea_level.keys():
        average_region_sea_level[date] = average_region_sea_level[date] - avg

    # plot
    plot_timelines([(average_region_sea_level, "mean sea level mediterranean sea", "blue")],
                   "mediterranean_sea_level", OUTPUT_DIRECTORY)

    # select all station that lie in the mediterranean sea and average their sea level for each clustering solution
    # and compare it to the grid data
    station_counter = 0
    selected_region_stations = []
    for station in STATIONS.values():
        if (station.longitude, station.latitude) in selected_coords:
            station_counter += 1
            selected_region_stations.append(station.id)
    # plot mediterranean stations
    land_gdf = geopandas.read_file(land_directory)
    land_gdf = land_gdf.explode("geometry", ignore_index=True)
    station_dict = {"id": [], "geometry": []}
    for station_id in selected_region_stations:
        current_station = STATIONS[station_id]
        station_dict["id"].append(current_station.id)
        station_dict["geometry"].append(shapely.geometry.Point(current_station.longitude, current_station.latitude))
    station_gdf = geopandas.GeoDataFrame(station_dict)
    ax = land_gdf.plot(color="burlywood", figsize=(40, 24), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    station_gdf.plot(ax=ax, color="red", marker=",", markersize=0.01, zorder=1)
    for i, txt in enumerate(station_gdf["id"]):
        ax.annotate(txt, (station_gdf["geometry"].iloc[i].x, station_gdf["geometry"].iloc[i].y), fontsize=0.1)
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, "selected_stations.pdf"))
    plt.close()
    logger.info(f"number of selected stations: {station_counter}")

    average_sea_level_per_radius = {}
    for radius in RADII:
        average_sea_level_per_date = {}
        for time_step in INTERVALS:
            start_year = time_step[0]
            end_year = time_step[1]
            file_path = (f"{CLUSTERING_INPUT_PATH}/{start_year}_{end_year}/solution_"
                         f"{radius}.json")
            with open(file_path) as file:
                current_solution = json.load(file)
            number_of_centers = 0
            for station_id in current_solution.keys():
                if int(station_id) in selected_region_stations:
                    current_station = STATIONS[int(station_id)]
                    time_series = current_station.timeseries.copy()
                    number_of_centers += 1
                    for date in time_series.keys():
                        if date >= start_year and date <= end_year:
                            if not date in average_sea_level_per_date.keys():
                                average_sea_level_per_date[date] = time_series[date]
                                average_sea_level_per_date[date] = time_series[date]
                            else:
                                average_sea_level_per_date[date] = average_sea_level_per_date[date] + \
                                                                   time_series[date]
            for date in average_sea_level_per_date.keys():
                if date >= start_year and date <= end_year:
                    average_sea_level_per_date[date] = average_sea_level_per_date[date] / number_of_centers
        # mean center the result
        average = 0
        for date in average_sea_level_per_date.keys():
            average += average_sea_level_per_date[date]
        average = average / len(average_sea_level_per_date)
        for date in average_sea_level_per_date.keys():
            average_sea_level_per_date[date] = average_sea_level_per_date[date] - average
        average_sea_level_per_radius[radius] = average_sea_level_per_date
    # plot
    rms = {}
    for radius in average_sea_level_per_radius.keys():
        plot_timelines([(average_region_sea_level, "mean sea level mediterranean sea", "blue"),
                        (average_sea_level_per_radius[radius], f"mean sea level mediterranean sea {radius}m", "green")],
                       f"mediterranean_sea_level_{radius}", OUTPUT_DIRECTORY)
        rms[radius] = calculate_RMS(average_sea_level_per_radius[radius], average_region_sea_level)
    # plot
    fig, ax = plt.subplots()
    ax.plot(*zip(*sorted(rms.items())), color="blue", zorder=1)
    plt.xlabel("radius")
    plt.ylabel("RMS")
    plt.savefig(f"{OUTPUT_DIRECTORY}mediterranean_sea_level_RMS.png")
    plt.close()

    logger.info(f"number of selected stations: {station_counter}")

    return


def which_hemisphere(clustering_path: str, stations_path: str, output_path: str):
    """
    Calculate the number of points that are in the northern/southern hemisphere in each clustering
    :param clustering_path:
    :param stations_path:
    :param output_path:
    :return:
    """
    metadata_path = os.path.join(output_path, "metadata.txt")
    hemispheres_path = os.path.join(output_path, "hemispheres.txt")
    # read stations
    stations = src.inner.tide_gauge_station.read_and_create_stations(stations_path, metadata_path)
    northern_per_radius = {}
    southern_per_radius = {}
    for radius in RADII:
        northern_hemisphere = 0
        southern_hemisphere = 0
        for time_step in INTERVALS:
            start_year = time_step[0]
            end_year = time_step[1]
            file_path = (f"{clustering_path}/{start_year}_{end_year}/solution_"
                         f"{radius}.json")
            with open(file_path) as file:
                current_solution = json.load(file)

            for station_id in current_solution.keys():
                station = stations[int(station_id)]
                if station.latitude > 0:
                    northern_hemisphere += 1
                else:
                    southern_hemisphere += 1
        avg_northern_hemisphere = northern_hemisphere / len(INTERVALS)
        avg_southern_hemisphere = southern_hemisphere / len(INTERVALS)
        northern_per_radius[radius] = avg_northern_hemisphere
        southern_per_radius[radius] = avg_southern_hemisphere

        with open(hemispheres_path, "a") as file:
            file.write(f"radius: {radius}\n")
            file.write(f"average number of stations in the northern hemisphere: {avg_northern_hemisphere}\n")
            file.write(f"average number of stations in the southern hemisphere: {avg_southern_hemisphere}\n\n")
    # plot
    fig, ax = plt.subplots()
    ax.plot(*zip(*sorted(northern_per_radius.items())), color="blue", zorder=1, label="northern hemisphere")
    ax.plot(*zip(*sorted(southern_per_radius.items())), color="red", zorder=2, label="southern hemisphere")
    plt.legend()
    plt.xlabel("radius")
    plt.ylabel("number of stations")
    plt.savefig(f"{output_path}hemispheres.png")
    plt.close()
    return


if __name__ == "__main__":
    clustering_directory = "../output/RMS/time_steps-vs-entire_timespan-ORAS5/time_steps"
    oras5_directory = "../data/Oras5"
    oras_dimensions_directory = "../data/oras5_gridDim/GLO-MFC_001_018_coordinates.nc"
    stations_directory = "../data/Meeresdaten_simuliert/filelist.txt"
    output_directory = "../output/RMS/time_steps-vs-entire_timespan-ORAS5/time_steps/evaluation"
    land_directory = "../data/ne_10m_land/ne_10m_land.shp"
    altimetry_directory = "../data/global_sea_level_altimetry/gsmsl_2023rel2_season_retained.txt"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # all_intervals = [(1954, 1964), (1964, 1974), (1974, 1984), (1984, 1994), (1994, 2004), (2004, 2014), (2014, 2024)]
    all_intervals = [(1958, 1968), (1968, 1978), (1978, 1988), (1988, 1998), (1998, 2008), (2008, 2018), (2018, 2028)]
    radii = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070,
             0.075, 0.080, 0.085, 0.090, 0.095, 0.100]
    time_steps_given = True
    init(oras5_directory, oras_dimensions_directory, output_directory, stations_directory, all_intervals, radii,
         clustering_directory)
    # evaluate_selected_regions(oras5_directory, oras_dimensions_directory)
    # calculate the number of points that are in the northern/southern hemisphere in each clustering
    # which_hemisphere(clustering_directory, stations_directory, output_directory)
    # logger.info("Calculating global mean sea level...")

    start(time_steps_given)
