import math
import os

import matplotlib.pyplot as plt
from geopandas import read_file
from loguru import logger

from src.inner import plot
from src.inner.plot import plot_current_year
from src.inner.tide_gauge_station import TideGaugeStation


def subtract_mean_from_timeseries(current_stations: {int: TideGaugeStation}):
    """
    Subtract the mean from each value in the timeseries for each tide gauge station, do not regard -99999 values,
    as they code a missing value
    :param current_stations:
    :return:
    """
    stations_without_data = []
    for station in current_stations.values():
        sum = 0
        amount_of_values = 0
        for date, sea_level in station.timeseries.items():
            if sea_level != -99999:
                sum += sea_level
                amount_of_values += 1
        if amount_of_values == 0:
            stations_without_data.append(station.id)
        else:
            mean = sum / amount_of_values
            for date, sea_level in station.timeseries.items():
                if sea_level != -99999:
                    station.timeseries_detrended_normalized[date] = sea_level - mean
                else:
                    station.timeseries_detrended_normalized[date] = -99999
    for station in stations_without_data:
        current_stations.pop(station)
    logger.info(f"Found {len(stations_without_data)} stations without data")
    logger.info(f"Now there are {len(current_stations)} stations in the dataset.")
    return current_stations


def average_amount_of_missing_data(current_stations: {int: TideGaugeStation}, metadata_path: str):
    """
    Calculate the average amount of missing data for all stations
    :return:
    """
    # Check how much data is missing in the timelines on average
    # Check if there are stations for which the timeline is complete
    total_missing_data = 0
    complete_timeseries = 0
    total_values = 0
    values_per_station = []
    earliest_date = float("inf")
    latest_date = 0
    for station in current_stations.values():
        completeness = True
        length_of_timeseries = 0
        for date, sea_level in station.timeseries.items():
            total_values += 1
            if sea_level == -99999:
                total_missing_data += 1
                completeness = False
            if date < earliest_date:
                earliest_date = date
            if date > latest_date:
                latest_date = date
            else:
                length_of_timeseries += 1
        values_per_station.append(length_of_timeseries)
        if completeness:
            complete_timeseries += 1
    if len(current_stations) != 0:
        total_missing_data = total_missing_data / len(current_stations)
        total_values = total_values / len(current_stations)
        with open(metadata_path, "a") as file:
            file.write(f"Number of stations with any data: {len(current_stations)}\n")
            file.write(f"Average amount of -99999 values: {total_missing_data}\n")
            file.write(f"Timeseries without -99999 values: {complete_timeseries}\n")
            file.write(f"Average amount of values in timeseries: {total_values}\n")
            file.write(f"Minimum amount of values in a timeseries: {min(values_per_station)}\n")
            file.write(f"Maximum amount of values in a timeseries: {max(values_per_station)}\n")
            file.write(
                f"Median amount of values in timeseries: {sorted(values_per_station)[len(values_per_station) // 2]}\n")
            file.write(f"Earliest date: {earliest_date}\n")
            file.write(f"Latest date: {latest_date}\n")
    else:
        with open(metadata_path, "a") as file:
            file.write(f"No stations in the dataset\n")

    return


def calculate_rms_difference_between_pairs_of_stations(station_a: TideGaugeStation, station_b: TideGaugeStation,
                                                       metadata_path: str):
    """
    Calculate the difference in sea level between pairs of stations (rms)
    If there is a gap in the data, ignore the data points in the gap
    Additionally calculate how much of station b is covered by station a and vice versa
    :return:
    """
    percentage_of_a_covered_by_b = 0
    total_a_values = 0
    percentage_of_b_covered_by_a = 0
    total_b_values = 0
    # RMS
    time_series_a = station_a.timeseries_detrended_normalized.copy()
    time_series_b = station_b.timeseries_detrended_normalized.copy()
    # calculate how many values there are in each timeseries that are not -99999
    gap_counter, total_a_values, total_b_values = check_overlap_and_remove_missing_data(time_series_a, time_series_b,
                                                                                        total_a_values, total_b_values)
    # check if there is something left to compare
    if len(time_series_a.keys()) == 0 or len(time_series_b.keys()) == 0:
        # logger.info(f"No data to compare between {station_a.id} and {station_b.id}")
        return 0, gap_counter, float("inf"), 0, 0
    # calculate the difference
    difference = 0
    for date in time_series_a.keys():
        difference += abs(time_series_a[date] - time_series_b[date]) * abs(time_series_a[date] - time_series_b[date])
    if difference != 0:
        difference = difference / len(time_series_a)
        difference = math.sqrt(difference)
    else:
        with open(metadata_path, "a") as file:
            file.write(f"No difference between {station_a.id} and {station_b.id} \n")
    overlap = len(time_series_a)

    # calculate percentage of a covered by b and vice versa
    percentage_of_a_covered_by_b = overlap / total_a_values * 100
    percentage_of_b_covered_by_a = overlap / total_b_values * 100

    return overlap, gap_counter, difference, percentage_of_a_covered_by_b, percentage_of_b_covered_by_a


def check_overlap_and_remove_missing_data(time_series_a, time_series_b, total_a_values, total_b_values):
    for date in time_series_a.keys():
        if time_series_a[date] != -99999:
            total_a_values += 1
    for date in time_series_b.keys():
        if time_series_b[date] != -99999:
            total_b_values += 1
    # remove gaps in the data from both timelines
    gap_counter = 0
    remove_from_a = []
    remove_from_b = []
    # check for dates that are in timeseries a, but not in timeseries b and vice versa
    for date in time_series_a.keys():
        if date not in time_series_b.keys():
            remove_from_a.append(date)
            gap_counter += 1
    for date in time_series_b.keys():
        if date not in time_series_a.keys():
            remove_from_b.append(date)
            gap_counter += 1
    # remove invalid values
    for date in remove_from_a:
        time_series_a.pop(date)
    for date in remove_from_b:
        time_series_b.pop(date)
    # now check for dates that do not have valid values in either of the timelines
    remove_from_a = []
    remove_from_b = []
    for date in time_series_a:
        if time_series_a[date] == -99999 or time_series_b[date] == -99999:
            remove_from_a.append(date)
            remove_from_b.append(date)
            gap_counter += 1
    # remove invalid values
    for date in remove_from_a:
        time_series_a.pop(date)
    for date in remove_from_b:
        time_series_b.pop(date)
    return gap_counter, total_a_values, total_b_values


def calculate_mae_difference_between_pairs_of_stations(station_a, station_b, metadata_path: str):
    """
    Calculate the difference in sea level between pairs of stations (mae)
    :param station_a:
    :param station_b:
    :return:
    """
    percentage_of_a_covered_by_b = 0
    total_a_values = 0
    percentage_of_b_covered_by_a = 0
    total_b_values = 0
    # MAE
    time_series_a = station_a.timeseries_detrended_normalized.copy()
    time_series_b = station_b.timeseries_detrended_normalized.copy()
    # calculate how many values there are in each timeseries that are not -99999
    gap_counter, total_a_values, total_b_values = check_overlap_and_remove_missing_data(time_series_a, time_series_b,
                                                                                        total_a_values, total_b_values)
    # check if there is something left to compare
    if len(time_series_a.keys()) == 0 or len(time_series_b.keys()) == 0:
        # logger.info(f"No data to compare between {station_a.id} and {station_b.id}")
        return 0, gap_counter, float("inf"), 0, 0
    # calculate the difference
    difference = 0
    for date in time_series_a.keys():
        difference += abs(time_series_a[date] - time_series_b[date])
    if difference != 0:
        difference = difference / len(time_series_a)

    else:
        with open(metadata_path, "w") as file:
            file.write(f"No difference between {station_a.id} and {station_b.id}/n")
    overlap = len(time_series_a)

    # calculate percentage of a covered by b and vice versa
    percentage_of_a_covered_by_b = overlap / total_a_values * 100
    percentage_of_b_covered_by_a = overlap / total_b_values * 100

    return overlap, gap_counter, difference, percentage_of_a_covered_by_b, percentage_of_b_covered_by_a


def calculate_difference_between_all_pairs_of_stations(current_stations: {int: TideGaugeStation},
                                                       metadata_path: str, rms: bool, mae: bool):
    """
    Calculate the difference in sea level between all pairs of stations (rms or mae)
    :param mae:
    :param rms:
    :param metadata_path:
    :param current_stations:
    :return:
    """
    diffs = {}
    amount_incomparable_timelines = 0
    amount_comparable_timelines = 0
    stations_ids = list(current_stations.keys())
    all_gaps = 0
    individual_gaps = []
    overlaps = []
    incomparable = {}
    number_of_pairs = 0
    number_of_stations = len(current_stations)
    for station_a_index in range(number_of_stations):
        for station_b_index in range(station_a_index + 1, number_of_stations):
            station_a = current_stations[stations_ids[station_a_index]]
            station_b = current_stations[stations_ids[station_b_index]]
            if station_a != station_b:
                number_of_pairs += 1
                if rms:
                    overlap, gap_counter, difference, percentage_a, percentage_b = (
                        calculate_rms_difference_between_pairs_of_stations(station_a, station_b, metadata_path))
                elif mae:
                    overlap, gap_counter, difference, percentage_a, percentage_b = (
                        calculate_mae_difference_between_pairs_of_stations(station_a, station_b, metadata_path))
                all_gaps += gap_counter
                overlaps.extend([overlap])
                individual_gaps.extend([gap_counter])
                if station_a.id not in diffs.keys():
                    diffs[station_a.id] = {}
                if station_b.id not in diffs.keys():
                    diffs[station_b.id] = {}
                diffs[station_a.id][station_b.id] = (percentage_b, difference)
                diffs[station_b.id][station_a.id] = (percentage_a, difference)
                if difference == float("inf"):
                    amount_incomparable_timelines += 1
                    incomparable[station_a.id] = station_b.id
                else:
                    amount_comparable_timelines += 1

    with open(metadata_path, "a") as file:
        file.write("\n")
        file.write(f"There are {number_of_pairs} pairs of timelines\n")
        file.write(f"Found {amount_incomparable_timelines} incomparable timelines\n")
        file.write(f"Found {amount_comparable_timelines} comparable timelines\n")
        if number_of_pairs != 0:
            file.write(f"Percentage of comparable timelines: {amount_comparable_timelines / number_of_pairs * 100}%\n")
            file.write(
                f"Percentage of incomparable timelines: {amount_incomparable_timelines / number_of_pairs * 100}%\n")
        file.write("\n")
        file.write(f"Found {all_gaps} gaps in the data\n")
        if number_of_pairs != 0:
            file.write(f"Average amount of gaps for a pair of timelines: {all_gaps / number_of_pairs}\n")
            file.write(f"Minimum amount of gaps for a pair of timelines: {min(individual_gaps)}\n")
            file.write(f"Maximum amount of gaps for a pair of timelines: {max(individual_gaps)}\n")
            file.write(
                f"Median amount of gaps for a pair of timelines: "
                f"{sorted(individual_gaps)[len(individual_gaps) // 2]}\n")
            file.write("\n")
        if number_of_pairs != 0:
            file.write(f"Average amount of overlap for a pair of timelines: {sum(overlaps) / number_of_pairs}\n")
            file.write(f"Minimum amount of overlap for a pair of timelines: {min(overlaps)}\n")
            file.write(f"Maximum amount of overlap for a pair of timelines: {max(overlaps)}\n")
            file.write(
                f"Median amount of overlap for a pair of timelines: {sorted(overlaps)[len(overlaps) // 2]}\n")

        all_differences = []
        for key in diffs.keys():
            for sub_key in diffs[key].keys():
                all_differences.append(diffs[key][sub_key][1])
        with open(metadata_path, "a") as file:
            file.write("\n")
            if len(all_differences) != 0:
                file.write(f"Overall {len(all_differences)} sea level rms-difference values \n")
                file.write(f"Average difference: {sum(all_differences) / len(all_differences)}\n")
                file.write(f"Minimum difference: {min(all_differences)} \n")
                file.write(f"Maximum difference: {max(all_differences)}\n")
                file.write(f"Median difference: {sorted(all_differences)[len(all_differences) // 2]}\n")
                file.write("\n")
            else:
                file.write("There are not enough values to compare\n")

    return diffs


def plot_timelines(current_stations: {int: TideGaugeStation}, plot_path: str, timesteps: [int]):
    """
    plot graph showing how many timelines are present at each date in the dataset
    :param plot_path:
    :param current_stations:
    :return:
    """
    date_amount = {}
    not_valid = {}
    amount_invalid = 0
    overall = {}
    for station in current_stations.values():
        for date in station.timeseries.keys():
            overall[date] = 1585
            if station.timeseries[date] != -99999:
                if date not in date_amount.keys():
                    date_amount[date] = 1
                else:
                    date_amount[date] = date_amount[date] + 1
            else:
                amount_invalid += 1
                if date not in not_valid.keys():
                    not_valid[date] = 1
                else:
                    not_valid[date] = not_valid[date] + 1
    ordered_dates = sorted(date_amount.items())
    x, y = zip(*ordered_dates)
    plt.plot(x, y)
    plt.xticks(timesteps)
    plt.xlabel("Date")
    plt.ylabel("Amount of timelines")
    plt.savefig(plot_path + "timelines.pdf")
    plt.show()

    ordered_missing_dates = sorted(not_valid.items())
    z, w = zip(*ordered_missing_dates)
    plt.plot(x, y)
    plt.plot(z, w, color="red")
    plt.xticks(timesteps)
    plt.xlabel("Date")
    plt.ylabel("Amount of timelines")
    plt.savefig(plot_path + "timelines+missing_data.pdf")
    plt.show()

    missing_data = {}
    for date in date_amount.keys():
        missing_data[date] = len(current_stations) - date_amount[date]

    ordered_overall = sorted(overall.items())
    b, c = zip(*ordered_overall)
    plt.plot(x, y)
    plt.plot(z, w, color="red")
    plt.plot(b, c, color="purple")
    plt.xticks(timesteps)
    plt.xlabel("Date")
    plt.ylabel("Amount of timelines")
    plt.savefig(plot_path + "timelines+missing_data_overall.pdf")
    plt.show()

    missing_data_ordered = sorted(missing_data.items())
    e, f = zip(*missing_data_ordered)
    plt.plot(e, f, color="red")
    plt.plot(x, y, color="blue")
    plt.xticks(timesteps)
    plt.xlabel("Date")
    plt.ylabel("Missing/Present timelines")
    plt.savefig(plot_path + "missing_data+timelines.pdf")
    plt.show()

    logger.info(f"Found {amount_invalid} -99999 values in the dataset")
    sorted_values = sorted(date_amount.values())
    logger.info(f"Number of dates: {len(date_amount.keys())}")
    logger.info(f"Average amount of timelines for a date: {sum(sorted_values) / len(sorted_values)}")
    logger.info(f"Minimum amount of timelines for a date: {sorted_values[0]}")
    logger.info(f"Maximum amount of timelines for a date: {sorted_values[-1]}")
    return


def remove_dates_before_and_after_threshold(start_year: int, end_year: int, current_stations: {int: TideGaugeStation},
                                            metadata_path: str):
    """
    Remove all dates before and after certain threshold
    :param metadata_path:
    :param end_year:
    :param start_year:
    :param current_stations:
    :return:
    """
    stations_removed_dates = {}
    for station in current_stations.values():
        # need to copy the timeseries, otherwise the original timeseries will be changed and this will not work for
        # the next timestep
        new_station = TideGaugeStation(station.id, station.name, station.latitude,
                                       station.longitude, station.timeseries.copy(),
                                       station.timeseries_detrended_normalized.copy(), None)
        for date in list(new_station.timeseries.keys()):
            if date < start_year or date > end_year:
                new_station.timeseries.pop(date)
        stations_removed_dates[new_station.id] = new_station
    # remove empty stations
    stations_to_remove = []
    removed_stations = 0
    for station in stations_removed_dates.values():
        if len(station.timeseries.keys()) == 0:
            stations_to_remove.append(station.id)
            removed_stations += 1
    with open(metadata_path, "a") as file:
        file.write(f"Removed {removed_stations} stations, because they have no values in the considered timespan\n")
    for station in stations_to_remove:
        stations_removed_dates.pop(station)
    with open(metadata_path, "a") as file:
        file.write(f"Now there are {len(stations_removed_dates)} stations in the dataset.\n")

    return stations_removed_dates


def plot_present_stations(current_stations: [TideGaugeStation], start_year: int, end_year: int, step: int):
    """
    Plot all stations that are present in a given year between 1822 and 2023
    :param current_stations:
    :param start_year:
    :param end_year:
    :param step:
    :return:
    """
    years = list(range(start_year, end_year, step))
    landmass_file_path = "../../data/ne_10m_land/ne_10m_land.shp"
    land_gdf = read_file(landmass_file_path)
    land_gdf = land_gdf.explode("geometry", ignore_index=True)
    station_ids = []
    for year in years:
        geometry = []
        station_ids = plot_current_year(current_stations, geometry, station_ids, land_gdf, year)

    return


def write_differences_to_file(differences: {}, path: str):
    """
    Write the differences between all pairs of stations to a file
    :param differences:
    :param path:
    :return:
    """
    sea_level = {}
    percentage = {}
    for key in differences.keys():
        for sub_key in differences[key].keys():
            if key not in sea_level.keys():
                sea_level[key] = {}
                percentage[key] = {}
            sea_level[key][sub_key] = differences[key][sub_key][1]
            percentage[key][sub_key] = differences[key][sub_key][0]
    save_differences_to_file(sea_level, path, "difference.txt")
    save_differences_to_file(percentage, path, "percentage.txt")
    return


def save_differences_to_file(diffs: {int: {int: float}}, current_output_dir: str, name: str):
    """
    Save the differences between all pairs of stations to a file
    :param diffs:
    :param current_output_dir:
    :param time_step:
    :return:
    """
    with open(f"{current_output_dir}/{name}", "w") as file:
        for key in diffs.keys():
            for sub_key in diffs[key].keys():
                file.write(f"{key} {sub_key} {diffs[key][sub_key]}\n")
    return


def read_differences_from_file(current_output_dir: str, name: str):
    """
    Read the differences between all pairs of stations from a file
    :param name:
    :param current_output_dir:
    :return:
    """
    diffs = {}
    with open(f"{current_output_dir}/{name}", "r") as file:
        for line in file:
            split_line = line.split(" ")
            station_a = int(split_line[0])
            station_b = int(split_line[1])
            difference = float(split_line[2])
            if station_a not in diffs.keys():
                diffs[station_a] = {}
            diffs[station_a][station_b] = difference
    return diffs


def remove_percentages(diffs_with_percentages: {str: {str: [float, float]}}):
    """
    Remove the percentages from the differences
    :param diffs_with_percentages:
    :return:
    """
    diffs = {}
    for key in diffs_with_percentages.keys():
        diffs[key] = {}
        for second_key in diffs_with_percentages[key].keys():
            diffs[key][second_key] = diffs_with_percentages[key][second_key][1]
    return diffs


def calculate_percentage_difference(diffs_with_percentages: {str: {str: [float, float]}}):
    """
    Extract the percentage of overlap between all pairs of stations
    :param diffs_with_percentages:
    :return:
    """
    percentages = {}
    for key in diffs_with_percentages.keys():
        percentages[key] = {}
        for second_key in diffs_with_percentages[key].keys():
            percentages[key][second_key] = diffs_with_percentages[key][second_key][0]
    return percentages


def calculate_time_series_difference(current_output_dir, filtered_stations, mae, rms):
    """
    Calculate the difference between all pairs of stations
    :param current_output_dir:
    :param filtered_stations:
    :param mae:
    :param rms:
    :return:
    """
    average_amount_of_missing_data(filtered_stations, os.path.join(current_output_dir, "current_stations.txt"))
    diffs_with_percentages = calculate_difference_between_all_pairs_of_stations(
        filtered_stations, os.path.join(current_output_dir, "current_stations.txt"), rms, mae)
    current_difference = remove_percentages(diffs_with_percentages)
    current_percentage = calculate_percentage_difference(diffs_with_percentages)
    write_differences_to_file(diffs_with_percentages, current_output_dir)
    plot.plot_all_stations(filtered_stations, current_output_dir)
    plot.plot_difference_histogram(diffs_with_percentages, current_output_dir)
    return current_difference, current_percentage
