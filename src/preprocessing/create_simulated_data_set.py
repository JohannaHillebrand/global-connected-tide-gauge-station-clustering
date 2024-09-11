import glob
import json
import os
import time
from multiprocessing import pool

import netCDF4
import numpy as np
from astropy.time import Time
from haversine import haversine
from tqdm import tqdm

import src.inner.tide_gauge_station


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


def find_closest_grid_point(quadruple):
    """
    For a given station, find the closest grid point in the oras5 dataset using the haversine formula
    :param all_longitudes:
    :param all_latitudes:
    :param current_station:
    :return:
    """
    all_latitudes, all_longitudes, current_station, sossheig = quadruple
    closest_point = float("inf")
    grid_point = None
    for index1, (lat_array, lon_array) in enumerate(zip(all_latitudes, all_longitudes)):
        for index2, (lat, lon) in enumerate(zip(lat_array, lon_array)):
            if (lat <= current_station.latitude + 10 and lat >= current_station.latitude - 10 and lon <=
                    current_station.longitude + 10 and lon >= current_station.longitude - 10):
                if sossheig.recordmask[index1, index2]:
                    continue
                if lon > 180.0 or lon < -180.0:
                    continue
                haversine_distance = haversine((current_station.latitude, current_station.longitude), (lat, lon))
                if haversine_distance < closest_point:
                    closest_point = haversine_distance
                    grid_point = (lat, lon)
                    index = (index1, index2)
    # print(f"closest distance: {closest_point}")
    # print(f"grid point: {grid_point}")
    # print(
    #     f"lats index: "
    #     f"{all_latitudes[station_to_index[current_station.id][0], station_to_index[current_station.id][1]]}")
    # print(
    #     f"lons index: "
    #     f"{all_longitudes[station_to_index[current_station.id][0], station_to_index[current_station.id][1]]}")
    # print(f"station: {current_station.latitude}, {current_station.longitude}")
    # print(f"index: {station_to_index[current_station.id]}")

    if grid_point is not None:
        current_station.latitude = grid_point[0]
        current_station.longitude = grid_point[1]
    else:
        return
    return [current_station, index]


def start(output_path: str):
    """
    start function for creating a simulated data set
    :return:
    """
    time1 = time.time()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # find closes grid point for each station
    stations = src.inner.tide_gauge_station.read_and_create_stations("../../data/rlr_monthly/filelist.txt",
                                                                     os.path.join(output_path, "metadata.txt"))
    for station in stations.values():
        station.timeseries = {}
        station.timeseries_detrended_normalized = {}
    # read one oras5 file to get the grid points
    oras5 = netCDF4.Dataset("../../data/Oras5/1958-1979/sossheig_control_monthly_highres_2D_195801_CONS_v0.1.nc")
    print(oras5.variables.keys())
    print(oras5.variables["nav_lat"])
    lats = oras5.variables["nav_lat"][:]
    lons = oras5.variables["nav_lon"][:]
    sossheig = oras5.variables["sossheig"][:].squeeze()
    counter = 0
    station_to_index = {}
    station_quadruples = []
    for station in tqdm(stations.values()):
        station_quadruples.append((lats, lons, station, sossheig))
        # counter += 1
        # if counter == 10:
        #     break

    with pool.Pool() as p:
        results = list(
            tqdm(p.imap(find_closest_grid_point, station_quadruples, chunksize=1), total=len(station_quadruples)))
        p.close()

    modified_stations = {}
    for result in tqdm(results):
        if result is not None:
            station = result[0]
            index = result[1]
            station_to_index[station.id] = index
            modified_stations[station.id] = station
    duplicates = {}
    items = list(station_to_index.items())
    for position1 in range(len(items)):
        for position2 in range(position1 + 1, len(items)):
            id1 = items[position1][0]
            index = items[position1][1]
            id2 = items[position2][0]
            index2 = items[position2][1]
            if index[0] == index2[0] and index[1] == index2[1]:
                if id1 not in duplicates:
                    duplicates[id1] = []
                duplicates[id1].append(id2)
    print(f"duplicates: {duplicates}")
    for stations_to_remove in duplicates.values():
        for station_id in stations_to_remove:
            if station_id in modified_stations.keys():
                modified_stations.pop(station_id)

    # create simulated data set
    directories = [x[0] for x in os.walk("../../data/Oras5")][1:]
    for directory in directories:
        print(directory)
        files = glob.glob(f"{directory}/*.nc")
        for file in files:
            oras5 = netCDF4.Dataset(file)
            sossheig = oras5.variables["sossheig"][:].squeeze()

            current_date_netcdf = netCDF4.num2date(oras5.variables["time_counter"][:],
                                                   oras5.variables["time_counter"].units,
                                                   only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            # this gives us an ERFA warning, because UTC began in 1960 and it is not "proper" to use this function for
            # dates before. However, it seems to function correctly.
            astropy_time_object = Time(current_date_netcdf[0], format="datetime", scale="utc")
            current_date = astropy_time_object.decimalyear

            stations_to_remove = []
            for station in modified_stations.values():
                try:
                    sea_level = sossheig[station_to_index[station.id][0], station_to_index[station.id][1]]
                    station.timeseries[current_date] = sea_level
                except:
                    stations_to_remove.append(station)
    print(f"stations to remove: {len(stations_to_remove)}")
    print(f"stations: {len(modified_stations)}")
    for station in stations_to_remove:
        if station.id in modified_stations.keys():
            modified_stations.pop(station.id)
    print(f"stations: {len(modified_stations)}")
    # print(stations)
    # write to file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_path = os.path.join(output_path, "data/")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(os.path.join(output_path, "filelist.txt"), "w") as file:
        for station in modified_stations.values():
            file.write(f"{station.id}; {station.latitude}; {station.longitude}; {station.name};, 0; 0; N\n")
            station.timeseries = dict(sorted(station.timeseries.items()))
            with open(os.path.join(data_path, f"{station.id}.rlrdata"), "w") as station_file:
                for key, value in station.timeseries.items():
                    station_file.write(f"{key}; {value}; 0; 000\n")
    with open(os.path.join(output_path, "station_id_to_index.json"), "w") as file:
        json.dump(station_to_index, file)
    time2 = time.time()
    print(f"total Time: {time2 - time1}")


if __name__ == "__main__":
    start("../output/create_simulated_dataset_test_parallelization/")
