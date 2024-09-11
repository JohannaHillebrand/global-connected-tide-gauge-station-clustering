import json
import os.path

import matplotlib
import numpy as np
import xarray
from matplotlib import pyplot as plt

import src.inner.tide_gauge_station

matplotlib.use('Cairo')
# set fontsize to 20
plt.rcParams.update({'font.size': 20})


def start(reconstructed_data: xarray.Dataset, stations: {int: src.inner.tide_gauge_station.TideGaugeStation},
          cluster_size: int, time_steps: [(int, int)], clustering_path: str, output_dir: str):
    """
    Take reconstructed grid of tide gauge data and compare the time series at the tide gauge stations that were not 
    used in the reconstruction to the reconstructed time series at the closest grid point.
    :param output_dir:
    :param reconstructed_data: xarray.Dataset
    :param stations: {int: TideGaugeStation}
    :param cluster_size: int
    :param time_steps: [(int, int)]
    :param clustering_path: str
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # read in closest grid points
    closest_grid_points = {}
    if os.path.exists(os.path.join(output_dir, "closest_lat_lon.json")):
        with open(os.path.join(output_dir, "closest_lat_lon.json"), "r") as file:
            closest_grid_points = json.load(file)
    else:
        print(f"File {output_dir}closest_lat_lon.json does not exist")
        return
    unused_station_ids = list(stations.keys()).copy()
    for time_step in time_steps:
        file_path = os.path.join(clustering_path, f"{time_step[0]}_{time_step[1]}", f"solution{cluster_size}.json")
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue
        # read clustering solution
        with open(file_path, "r") as file:
            clustering_solution = json.load(file)
        # find tide gauge stations that are not in the clustering solution, but have data for the current period
        for center in clustering_solution:
            if int(center) in unused_station_ids:
                unused_station_ids.remove(int(center))
    unused_stations_with_data = []
    for station_id in unused_station_ids:
        current_station = stations[station_id]
        present = False
        for date in current_station.timeseries.keys():
            if 1992 <= date <= 2023:
                present = True
                break
        if present:
            unused_stations_with_data.append(current_station.id)
    print(f"Removed {len(stations) - len(unused_stations_with_data)} station")
    print(f"Number of unused stations with data: {len(unused_stations_with_data)}")
    # for each station find the closest grid point
    if len(unused_stations_with_data) == 0:
        print("No unused stations with data")
        return
    # take the 10 unused stations with the longest time series after 1992
    unused_stations_with_data = sorted(
        unused_stations_with_data,
        key=lambda station: sum(1992 <= current_date <= 2023 for current_date in stations[station].timeseries.keys()),
        reverse=True
    )[:10]
    print(unused_stations_with_data)
    for station_id in unused_stations_with_data:
        current_station = stations[station_id]
        closest_lat, closest_lon = closest_grid_points[str(station_id)]
        closest_lon = float(closest_lon)
        closest_lat = float(closest_lat)
        closest_data = reconstructed_data.sel(lat=closest_lat, lon=closest_lon).sla.values
        # compare the time series at the tide gauge station to the time series at the closest grid point
        station_data = current_station.timeseries_detrended_normalized.copy()
        to_remove = []
        # mean center both time series
        mean = 0
        for date in station_data.keys():
            if station_data[date] != -99999:
                mean += station_data[date]
        mean /= len(station_data)
        for date in station_data.keys():
            if station_data[date] != -99999:
                station_data[date] -= mean
        mean = np.mean(closest_data)
        for element in closest_data:
            element -= mean

        for date in station_data.keys():
            if date < 1992 or date > 2023:
                to_remove.append(date)
            if station_data[date] == -99999:
                to_remove.append(date)
        for date in to_remove:
            if date in station_data:
                del station_data[date]
        for date in station_data.keys():
            # convert mm to m
            station_data[date] = station_data[date] / 1000
        # plot the comparison
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.plot(station_data.keys(), station_data.values(), label="Station", color="teal")
        ax.plot(reconstructed_data.time.values, closest_data, label="Closest grid point", color="firebrick")
        ax.set_title(f"Station {station_id}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Sea level anomaly [m]")
        ax.legend()
        plt.savefig(os.path.join(output_dir, f"station_{station_id}.svg"), dpi=400)
        plt.close()
        # calculate the RMS
        rms = 0
        for date in station_data.keys():
            closest_date = np.inf
            for date_reconstructed in reconstructed_data.time.values:
                if abs(date - date_reconstructed) < abs(date - closest_date):
                    closest_date = date_reconstructed
            if abs(closest_date - date) > 0.5:
                print(f"Date difference too large: {date}, {closest_date}")
                continue
            if station_data[date] == -99999:
                continue
            # print(f"date: {date}, closest_date: {closest_date}")
            rms += (station_data[date] - float(
                reconstructed_data.sel(lat=closest_lat, lon=closest_lon, time=closest_date).sla.values)) ** 2
        rms = (rms / len(station_data)) ** 0.5
        # write to file
        with open(os.path.join(output_dir, f"rms_stations.txt"), "a") as file:
            file.write(f"{station_id}\t{rms}\n")
        print(f"RMS for station {station_id}: {rms}")
