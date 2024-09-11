import json
import math
import os.path
from datetime import datetime, timedelta

import cftime
import matplotlib
import numpy as np
import pandas as pd
import xarray
import xarray as xr
from cartopy import crs as ccrs
from haversine import haversine
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.inner import tide_gauge_station

# use cairo
matplotlib.use("cairo")
plt.rcParams.update({'font.size': 20})


def read_altimetry_data(altimetry_data_path: str):
    """
    read altimetry data from the given path
    :param altimetry_data_path:
    :return:
    """
    first_load = True
    data_list = [f for f in os.listdir(altimetry_data_path) if f.endswith('.nc')]
    data_list.sort()
    for data in tqdm(data_list):
        file_path = os.path.join(altimetry_data_path, data)
        if os.path.exists(file_path):
            with xr.open_dataset(file_path) as current_dataframe:
                if first_load:
                    complete_dataframe = current_dataframe
                    first_load = False
                else:
                    complete_dataframe = xr.concat([complete_dataframe, current_dataframe], dim="time")

        else:
            logger.warning(f"File {file_path} does not exist")
    return complete_dataframe


def weight_altimetry_data(complete_dataframe):
    """
    weight the altimetry data by the cosine of the latitude to adjust for the area of the grid cells
    :param complete_dataframe:
    :return:
    """
    weighted_dataframe = complete_dataframe.copy()
    # weight each data point (sla) by the cosine of the latitude at the respective point
    if "latitude" not in complete_dataframe.coords:
        latitude_weights = np.cos(np.deg2rad(complete_dataframe.lat))

    else:
        latitude_weights = np.cos(np.deg2rad(complete_dataframe.latitude))
    # latitude_weights = np.sqrt(latitude_weights)
    # broadcast weight to the shape of the data
    weights = latitude_weights * xr.ones_like(complete_dataframe.sla)
    weighted_dataframe["weighted_sla"] = complete_dataframe.sla * weights
    # range for the weighted sla values
    sla_min = weighted_dataframe.weighted_sla.min().values
    sla_max = weighted_dataframe.weighted_sla.max().values
    logger.info(f"sla min: {sla_min}, sla max: {sla_max}")
    return weighted_dataframe


def create_input_matrix(weighted_dataframe_altimetry: xr.Dataset):
    """
    Turn the weighted data into a matrix where the there is a row for every stations time series and every column is
    a time step and an index dict that maps the latitude and longitude values to the row of the matrix
    :param weighted_dataframe_altimetry:
    :return:
    """
    # Get the unique latitude and longitude values
    logger.info("Creating input matrix")
    unique_latitudes = weighted_dataframe_altimetry.latitude.values
    unique_longitudes = weighted_dataframe_altimetry.longitude.values
    number_unique_latitudes = len(unique_latitudes)
    number_unique_longitudes = len(unique_longitudes)
    number_time_steps = len(weighted_dataframe_altimetry.time)

    # Preallocate the input matrix
    input_matrix = np.full((number_unique_latitudes * number_unique_longitudes, number_time_steps), np.nan)
    index_dict = {}

    # Flatten the latitude and longitude arrays for vectorized operations
    lat_lon_pairs = np.array(np.meshgrid(unique_latitudes, unique_longitudes)).T.reshape(-1, 2)

    # Vectorized selection of data
    idx = 0
    for ind, (lat, lon) in tqdm(enumerate(lat_lon_pairs)):
        data = weighted_dataframe_altimetry.sel(latitude=lat, longitude=lon).weighted_sla.values
        if not np.isnan(data).any():
            input_matrix[idx, :] = data
            # in which row is the data for this lon and lat stored
            index_dict[(lat, lon)] = idx
            idx += 1

    # Remove rows with NaN values from matrix
    input_matrix = input_matrix[~np.isnan(input_matrix).any(axis=1)]
    return input_matrix, index_dict


def perform_pca(input_matrix: np.ndarray, out_dir: str, number_of_components: int):
    """
    Perform PCA on the input matrix to reduce the dimensionality and extract the EOFs and PCs
    :param number_of_components:
    :param out_dir:
    :param input_matrix:
    :return:
    """
    # use scipy to perform PCA
    pca = PCA(number_of_components, svd_solver='full')
    pc = pca.fit_transform(input_matrix.T)
    # extract the EOFs and PCs
    eof = pca.components_.T
    latent = pca.explained_variance_ratio_
    print(f"explained variance: {sum(latent)}")
    with open(os.path.join(out_dir, "PCA.txt"), "w") as file:
        file.write("explained variance:\n")
        file.write("\n".join([str(value) for value in latent]))
        # sum up explained variance ratio
        file.write(f"\nsum of explained variance: {np.sum(latent)}")
        file.write("\nEOFs:\n")
        file.write(f"{eof.shape}")
        file.write("\nPCs:\n")
        file.write(f"{pc.shape}")
    return eof, pc


def make_dataset_out_of_eofs_and_plot(eof, index_dict, weighted_dataframe_altimetry, out_dir: str, pc):
    """
    Create a new xarray dataset with the EOF data
    :param out_dir:
    :param eof:
    :param index_dict:
    :param weighted_dataframe_altimetry:
    :return:
    """
    # Get unique latitudes and longitudes
    unique_latitudes = np.unique(weighted_dataframe_altimetry.latitude)
    unique_longitudes = np.unique(weighted_dataframe_altimetry.longitude)
    number_unique_latitudes = len(unique_latitudes)
    number_unique_longitudes = len(unique_longitudes)
    # Create 3-D array for EOF data using float32 to save memory
    eof_data = np.full((number_unique_latitudes, number_unique_longitudes, eof.shape[1]), np.nan, dtype=np.float32)
    # Create mappings for faster lookups
    lat_to_index = {lat: i for i, lat in enumerate(unique_latitudes)}
    lon_to_index = {lon: i for i, lon in enumerate(unique_longitudes)}
    # Determine if index_dict is zero-based or one-based
    min_index = min(index_dict.values())
    index_offset = 0 if min_index == 0 else -1
    # Fill in the data from the EOFs into the 3-D array
    for (lat, lon), index in index_dict.items():
        try:
            lat_idx = lat_to_index[lat]
            lon_idx = lon_to_index[lon]
            eof_data[lat_idx, lon_idx, :] = eof[index + index_offset, :]
        except KeyError as e:
            logger.warning(f"KeyError: {e}. Lat: {lat}, Lon: {lon} not found in mappings.")
        except IndexError as e:
            logger.warning(
                f"IndexError: {e}. Index: {index + index_offset} is out of bounds for eof array with shape "
                f"{eof.shape}.")

    # Create PC array
    pc_data = np.full(pc.shape, np.nan, dtype=np.float32)
    for i in range(pc.shape[1]):
        pc_data[:, i] = pc[:, i]

    # Create the Dataset
    eof_dataset = xr.Dataset(
        coords={
            "latitude": unique_latitudes,
            "longitude": unique_longitudes,
            "pca": [f"EOF_{i}" for i in range(eof_data.shape[2])],
            "time": weighted_dataframe_altimetry.time.values
        }
    )
    # Add the 'eof' variable to the Dataset
    eof_dataset["PAT"] = (["latitude", "longitude", "eof"], eof_data)
    # add the PCs to the dataset
    eof_dataset["pc_values"] = (["time", "eof"], pc_data)

    for i in range(eof.shape[1]):
        data = eof_dataset.eof_values.isel(eof=i)
        fig = plt.figure(figsize=(50, 25))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Plot the DataArray
        data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', add_colorbar=True)
        # Add coastlines and gridlines
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        # save
        plt.savefig(os.path.join(out_dir, f"EOF_{i}.svg"))
        plt.close(fig)
    # save as netcdf4 file for further processing
    eof_dataset.to_netcdf(os.path.join(out_dir, "EOFs.nc"))
    return eof_dataset


def remove_weight_and_normalize_eof(eof, index_dict):
    """
    remove the weight from the EOFs (cosine of the latitude)
    :param eof:
    :param index_dict:
    :return:
    """
    for (lat, lon), index in index_dict.items():
        eof[index, :] /= np.cos(np.deg2rad(lat))
    # # normalize the EOFs
    # eof = eof / np.linalg.norm(eof, axis=0)
    return eof


def apply_gia_to_tide_gauge_data(stations: {int: tide_gauge_station.TideGaugeStation}):
    """
    apply GIA (Glacial Isostatic adjustments) to the tide gauge data
    :param stations:
    :return:
    """
    for station in stations.values():
        pass
    return stations


def find_clustering_for_date(clustering_path: str, date: float, file_name: str):
    """
    find the appropriate clustering for the date
    :param file_name:
    :param clustering_path:
    :param date:
    :return:
    """
    clustering = {}
    directories = [f for f in os.listdir(clustering_path) if os.path.isdir(os.path.join(clustering_path, f))]
    for directory in directories:
        try:
            start_year = int(directory.split("_")[0])
            end_year = int(directory.split("_")[1])
            if start_year <= date <= end_year:
                if os.path.exists(os.path.join(clustering_path, directory, file_name)):
                    with open(os.path.join(clustering_path, directory, file_name), "r") as json_file:
                        clustering = json.load(json_file)
                else:
                    logger.warning(f"File {file_name} not found in {os.path.join(clustering_path, directory)}")
        except ValueError as e:
            continue
    if clustering == {}:
        logger.warning(f"No clustering found for date {date}")
    return clustering


def is_leap_year(year: int):
    """
    Check if the given year is a leap year
    :param year:
    :return:
    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def convert_decimal_date_to_datetime(decimal_date: float):
    """"
    Convert decimal date to datetime
    :param decimal_date:
    :return:
    """
    year = int(decimal_date)
    fractional_part = decimal_date - year
    days_in_year = 366 if is_leap_year(year) else 365
    day_of_year = int(fractional_part * days_in_year)
    jan_first = datetime(year, 1, 1)
    date = jan_first + timedelta(days=day_of_year)
    return date


def compare_pcs(alpha_for_date: dict, eof_dataset: xarray.Dataset, out_dir: str, number_of_eofs: int):
    """
    Compare the PCs for the reconstructed data with the PCs from the EOFs
    :param alpha_for_date:
    :param eof_dataset:
    :param out_dir:
    :return:
    """
    # sort the dict alpha_for_date by date and create a list of the values
    sorted_alpha = {k: v for k, v in sorted(alpha_for_date.items(), key=lambda item: item[0])}
    # get the PCs and the dates from the eof dataset
    eof_pcs = np.array(eof_dataset.PC[0:number_of_eofs, :])
    eof_dates = np.array(eof_dataset.time.values)
    # plot the PCs
    for i in range(number_of_eofs):
        # plot the PCs
        fig, ax = plt.subplots(figsize=(10, 5))
        # pcs are in shape (number_of_eofs, number_of_dates)
        ax.plot(eof_dates, eof_pcs[i], color="blue", linewidth=2,
                label="Altimetry")
        # [pc[i][0] for pc in sorted_pcs]
        ax.plot(list(sorted_alpha.keys()), [pc[i][0] for pc in list(sorted_alpha.values())], color="red",
                label="Reconstructed")
        ax.set_xlabel("Date")
        ax.set_ylabel("Global mean sea level")
        # set x-ticks to only show every 5 years
        ax.set_xticks([year for year in
                       range(int(min(list(sorted_alpha.keys()))), int(max(list(sorted_alpha.keys()))),
                             5)])
        ax.legend()
        plt.savefig(os.path.join(out_dir, f"PCs_{i}.svg"), dpi=600)
        plt.close(fig)

        #

    pass


def reconstruct_data(eof_dataset: xr.Dataset, stations: {int: tide_gauge_station.TideGaugeStation},
                     clustering_path: str, file_name: str, out_dir: str,
                     number_of_eofs: int, start_date: int, end_date: int):
    """

    :param end_date:
    :param start_date:
    :param number_of_eofs:
    :param weighted_dataframe_altimetry:
    :param file_name:
    :param eof_dataset:
    :param stations:
    :param clustering_path:
    :param out_dir:
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # The altimeter data is in meters, the stations are in mm - convert the stations to meters
    # check if there are enough tide gauge stations at a given date (more than eofs)
    # get the EOFs from the dataset
    # eofs = np.array([eof_dataset.PAT.isel(pca=i) for i in range(number_of_eofs)])
    # collect all dates for which there is any PSMSL data available
    all_dates = []
    for station in stations.values():
        for date in station.timeseries.keys():
            if date not in all_dates:
                all_dates.append(date)
    # sort the dates
    all_dates.sort()
    # plot_eofs(eof_dataset, out_dir, number_of_eofs)

    # assign each tide gauge station to the closest lat and lon in the altimetry dataset
    closest_lat_long_for_station = assign_stations_to_grid_points(eof_dataset, out_dir, stations,
                                                                  name="closest_lat_lon")

    # for each date in time take all stations that are centers at that point in time and use the EOFs to approximate
    # the PCs with linear least squares regression
    est_pc_for_date = {}
    logger.info(f"Reconstructing data for dates between {start_date} and {end_date}")
    for date in tqdm(all_dates):
        if date <= start_date:
            continue
        if date > end_date:
            continue
        # find the clustering for the date
        clustering = find_clustering_for_date(clustering_path, date, file_name)
        if clustering:
            # get the stations that are centers at that point in time
            centers = [int(center) for center in clustering.keys()]
            # get station objects for the centers (have id, lat, long, timeseries)
            current_stations = {station: stations[station] for station in centers}
            # filter the stations for the current date, such that only centers that have valid data are left
            current_stations, centers = filter_centers_for_date(centers, closest_lat_long_for_station, current_stations,
                                                                date)

            # check if there are more centers left than the number of eofs that are considered, otherwise skip the date
            if len(current_stations) < number_of_eofs:
                logger.warning(f"Not enough stations for date {date}")
                continue
            # create a vector with the observed data from the stations that are centers at that point in time
            # TODO: fill de vector with the difference between each pair of time steps
            tide_gauges = np.array(np.zeros((len(current_stations), 1)))
            for i, station in enumerate(current_stations.values()):
                if date in station.timeseries_detrended_normalized.keys():
                    tide_gauges[i] = (
                                         station.timeseries_detrended_normalized[
                                             date]) / 1000  # convert the tide gauge data to meters (from mm)

            # use only the eof-values that are at the grid point of a center
            reduced_eofs = np.array(np.zeros((len(centers), number_of_eofs)))
            for i, station in enumerate(current_stations.keys()):
                lat, lon = closest_lat_long_for_station[station]
                lat = float(lat)
                lon = float(lon)
                try:
                    reduced_eofs[i, :] = eof_dataset.sel(lat=lat, lon=lon).PAT.values[0:number_of_eofs]
                except KeyError as e:
                    logger.warning(f"KeyError: {e}. Lat: {lat}, Lon: {lon} not found in dataset.")

            # assume error for the tide gauge data
            sigma_squared = 1.0
            error_covariance_matrix = sigma_squared * np.eye(len(tide_gauges))
            inverse_error_covariance_matrix = np.linalg.inv(error_covariance_matrix)

            # weight tide gauges with inverse of the error covariance matrix
            weighted_tide_gauges = np.dot(inverse_error_covariance_matrix, tide_gauges)
            # least squares regression to estimate the PCs
            coefficients, residuals, rank, singular_values = np.linalg.lstsq(reduced_eofs, weighted_tide_gauges,
                                                                             rcond=None)
            est_pc_for_date[date] = coefficients
        else:
            logger.warning(f"No clustering found for date {date}")
    # TODO: change the reconstruction to use the difference between each pair of time steps to reconstruct the data

    # compare PCs for the reconstructed data with the PCs from the EOFs
    # compare_pcs(est_pc_for_date, eof_dataset, out_dir, number_of_eofs)
    # now reconstruct the data with the estimated pc values for every point in time
    # H_r = EOF(x,y)_m * alpha(t)_r

    data_arrays = []
    for date in est_pc_for_date.keys():
        # create field for the reconstructed data
        summed_h_r = np.zeros((eof_dataset.PAT.shape[1], eof_dataset.PAT.shape[2]))
        # sum up the H_r values for every EOF
        for i in range(number_of_eofs):
            current_eof = np.array(eof_dataset.PAT.isel(pca=i))
            current_alpha = est_pc_for_date[date][i]
            current_h_r = current_eof * current_alpha
            summed_h_r = np.sum([summed_h_r, current_h_r], axis=0)
        data_array = xr.DataArray(summed_h_r, dims=["lat", "lon"],
                                  coords={"lat": eof_dataset.lat, "lon": eof_dataset.lon,
                                          "time": date},
                                  name="sla")
        data_arrays.append(data_array)
    reconstructed_dataset = xr.concat(data_arrays, dim="time")
    reconstructed_dataset = reconstructed_dataset.to_dataset(name="sla")
    # save the reconstructed dataset as netcdf file

    return reconstructed_dataset


def filter_centers_for_date(centers, closest_lat_long_for_station, current_stations, date):
    """
    Filter the centers for the date, such that only centers that have data for the date in question are left
    do not use tide gauge stations that have -99999 for the current date
    :param next_date:
    :param centers:
    :param closest_lat_long_for_station:
    :param current_stations:
    :param date:
    :return:
    """
    stations_to_remove = []
    for i, station in enumerate(current_stations.values()):
        if date not in station.timeseries.keys():
            stations_to_remove.append(station.id)
        elif station.timeseries[date] == -99999:
            stations_to_remove.append(station.id)
        if station.id not in closest_lat_long_for_station.keys():
            stations_to_remove.append(station.id)
    for station in stations_to_remove:
        if station in current_stations.keys():
            current_stations.pop(station)
        if station in centers:
            centers.remove(station)
    return current_stations, centers


def assign_stations_to_grid_points(eof_dataset, out_dir, stations, name: str):
    """
    Assign the tide gauge stations to the closest lat and lon in the altimetry dataset
    :param eof_dataset: 
    :param out_dir: 
    :param stations: 
    :return: 
    """""
    if not os.path.exists(os.path.join(out_dir, f"{name}.json")):
        # get the closest lat and long for each station from the dataset
        logger.info(f"Finding closest lat and lon for each station")
        closest_lat_long_for_station = {}
        not_assigned_counter = 0
        for station in tqdm(stations.values()):
            station_lat = station.latitude
            station_lon = station.longitude
            # find the closest lat and lon in the eof dataset to the station lat and lon where the data is not none
            closest_lat, closest_lon = get_closest_grid_point(station.id, station_lat, station_lon, eof_dataset,
                                                              out_dir)
            if closest_lat is not None and closest_lon is not None:
                closest_lat_long_for_station[station.id] = (closest_lat, closest_lon)
            else:
                not_assigned_counter += 1
        if not_assigned_counter > 0:
            logger.info(f"Could not assign a closest lat and lon for {not_assigned_counter} stations")
        for key, value in closest_lat_long_for_station.items():
            # turn np.float32 (value) into float for json serialization
            closest_lat_long_for_station[key] = tuple(map(float, value))
        with open(os.path.join(out_dir, f"{name}.json"), "w") as json_file:
            json.dump(closest_lat_long_for_station, json_file)
    else:
        with open(os.path.join(out_dir, f"{name}.json"), "r") as json_file:
            closest_lat_long_for_station_json = json.load(json_file)
        # make id to int and lat and lon to float
        closest_lat_long_for_station = {int(k): tuple(map(float, v)) for k, v in
                                        closest_lat_long_for_station_json.items()}
    return closest_lat_long_for_station


def convert_cftime_to_datetime(cftime_date):
    """
    Convert cftime date to pandas Timestamp.
    :param cftime_date:
    :return:
    """
    if isinstance(cftime_date, cftime.datetime):
        # Convert cftime datetime to a pandas Timestamp
        return pd.Timestamp(cftime_date.strftime('%Y-%m-%dT%H:%M:%S'))
    else:
        return pd.Timestamp(cftime_date)


def calculate_and_compare_global_sea_level(out_dir: str, reconstructed_dataset: xr.Dataset,
                                           global_sea_level_altimetry_for_date,
                                           limited_altimetry_global_sea_level):
    """
    Calculate the global sea level from the reconstructed data and compare it to the altimetry data
    :param pca_dataset:
    :param number_of_eofs:
    :param out_dir:
    :param reconstructed_dataset:
    :param weighted_altimetry_dataset:
    :return:
    """
    # TODO: compare reconstructed sea level only with the m eofs and pcs
    weighted_reconstructed_sea_level = weight_altimetry_data(reconstructed_dataset)
    global_sea_level_for_date = {}
    for date in weighted_reconstructed_sea_level.time.values:
        data = np.array(weighted_reconstructed_sea_level.sel(time=date).weighted_sla.values)
        # calculate the mean sea level for where the value is not nan
        # sum_global_reconstructed_sea_level = 0
        # no_of_values = 0
        # for i in range(len(data)):
        #     for j in range(len(data[i])):
        #         if not np.isnan(data[i][j]):
        #             sum_global_reconstructed_sea_level += data[i][j]
        #             no_of_values += 1
        # global_mean_sea_level = sum_global_reconstructed_sea_level / no_of_values
        global_mean_sea_level = np.mean(data[~np.isnan(data)])
        global_sea_level_for_date[date] = global_mean_sea_level
        #
        # global_mean_sea_level = np.mean(data[~np.isnan(data)])
        # global_sea_level_for_date[date] = global_mean_sea_level
    # sort
    global_sea_level_for_date = {k: v for k, v in sorted(global_sea_level_for_date.items(), key=lambda item: item[0])}

    # sort
    global_sea_level_for_date = {k: v for k, v in sorted(global_sea_level_for_date.items(), key=lambda item: item[0])}
    # mean center over all time steps
    mean_reconstructed_sea_level = np.mean(list(global_sea_level_for_date.values()))
    for date in global_sea_level_for_date.keys():
        global_sea_level_for_date[date] -= mean_reconstructed_sea_level
    mean_altimetry_sea_level = np.mean(list(global_sea_level_altimetry_for_date.values()))
    for date in global_sea_level_altimetry_for_date.keys():
        global_sea_level_altimetry_for_date[date] -= mean_altimetry_sea_level
    mean_limited_altimetry_sea_level = np.mean(list(limited_altimetry_global_sea_level.values()))
    for date in limited_altimetry_global_sea_level.keys():
        limited_altimetry_global_sea_level[date] -= mean_limited_altimetry_sea_level

    # Calculate RMS between the reconstructed and altimetry data
    # for each date in altimetry data find the closest date in the reconstructed data
    rms = 0
    with open(os.path.join(out_dir, "dates_global_sea_level.txt"), "w") as dates_file:
        dates_file.write("Reconstructed dates:\n")
        dates_file.write("\n".join([str(date) for date in global_sea_level_for_date.keys()]))
        dates_file.write("\nAltimetry dates:\n")
        dates_file.write("\n".join([str(date) for date in global_sea_level_altimetry_for_date.keys()]))
    for date in global_sea_level_altimetry_for_date.keys():
        # calculate rms from 1993-2022
        if date < 1993 or date > 2022:
            continue
        closest_date = min(global_sea_level_for_date.keys(), key=lambda x: abs(x - date))
        rms += (global_sea_level_for_date[closest_date] - global_sea_level_altimetry_for_date[date]) ** 2
    rms = np.sqrt(rms / len(global_sea_level_altimetry_for_date))
    logger.info(f"RMS: {rms}")
    with open(os.path.join(out_dir, "rms.txt"), "w") as rms_file:
        rms_file.write(f"RMS: {rms}")

    # max and min values for the y-axis
    min_reconstructed = min(list(global_sea_level_for_date.values()))
    max_reconstructed = max(list(global_sea_level_for_date.values()))
    min_altimetry = min(list(global_sea_level_altimetry_for_date.values()))
    max_altimetry = max(list(global_sea_level_altimetry_for_date.values()))
    logger.info(f"min reconstructed: {min_reconstructed}, max reconstructed: {max_reconstructed}")
    logger.info(f"min altimetry: {min_altimetry}, max altimetry: {max_altimetry}")

    with open(os.path.join(out_dir, "dates.txt"), "w") as date_file:
        date_file.write("Reconstructed dates 1:\n")
        date_file.write("\n".join([str(date) for date in list(global_sea_level_for_date.keys())]))
        date_file.write("\nReconstructed dates 2:\n")
        # date_file.write("\n".join([str(date) for date in reconstructed_dates]))
        date_file.write("\nAltimetry dates:\n")
        date_file.write("\n".join([str(date) for date in global_sea_level_altimetry_for_date.keys()]))
        date_file.write("\nAltimetry dates 2:\n")
        # date_file.write("\n".join([str(date) for date in global_sea_level_altimetry_dates]))

    # plot global sea level and altimetry data for date
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(list(global_sea_level_for_date.keys()), list(global_sea_level_for_date.values()), color="firebrick",
            linewidth=4,
            label="Reconstructed")
    ax.plot(list(limited_altimetry_global_sea_level.keys()), list(limited_altimetry_global_sea_level.values()),
            color="goldenrod", linewidth=4, label="Limited altimetry")
    ax.plot(list(global_sea_level_altimetry_for_date.keys()), list(global_sea_level_altimetry_for_date.values()),
            color="teal", linewidth=4, label="Altimetry")
    ax.set_xlabel("Date")
    ax.set_ylabel("Global mean sea level")
    ax.set_xticks([year for year in
                   range(int(min(list(global_sea_level_for_date.keys()))),
                         int(max(list(global_sea_level_for_date.keys()))),
                         5)])
    ax.legend()
    plt.savefig(os.path.join(out_dir, "global_mean_sea_level.svg"))
    plt.close(fig)

    # only reconstructed data
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(list(global_sea_level_for_date.keys()), list(global_sea_level_for_date.values()), color="red", linewidth=4,
            label="Reconstructed")
    ax.set_xlabel("Date")
    ax.set_ylabel("Global mean sea level")
    ax.set_xticks([year for year in
                   range(int(min(list(global_sea_level_for_date.keys()))),
                         int(max(list(global_sea_level_for_date.keys()))),
                         5)])
    ax.legend()
    plt.savefig(os.path.join(out_dir, "global_mean_sea_level_reconstructed.svg"))
    plt.close(fig)
    return rms


def calculate_global_mean_sea_level_for_altimetry_data(number_of_eofs, pca_dataset, weighted_altimetry_dataset):
    limited_altimetry_global_sea_level = {}
    data_arrays = []
    for date in pca_dataset.time.values:
        sum_eofs = np.array(np.zeros((720, 1440)))
        for i in range(number_of_eofs):
            pc = pca_dataset.sel(time=date).PC.isel(pca=i).values
            eof = pca_dataset.PAT.isel(pca=i).values
            reconstructed_data = np.dot(eof, pc)
            sum_eofs = np.sum([sum_eofs, reconstructed_data], axis=0)
        # turn into data set
        data_array = xr.DataArray(sum_eofs, dims=["lat", "lon"],
                                  coords={"lat": pca_dataset.lat, "lon": pca_dataset.lon,
                                          "time": date},
                                  name="sla")
        data_arrays.append(data_array)
    limited_altimetry_dataset = xr.concat(data_arrays, dim="time")
    limited_altimetry_dataset = limited_altimetry_dataset.to_dataset(name="sla")
    # weight
    weighted_limited_altimetry_dataset = weight_altimetry_data(limited_altimetry_dataset)
    for date in weighted_limited_altimetry_dataset.time.values:
        data = np.array(weighted_limited_altimetry_dataset.sel(time=date).weighted_sla.values)
        # calculate the mean sea level for where the value is not nan
        global_mean_sea_level = np.mean(data[~np.isnan(data)])
        limited_altimetry_global_sea_level[date] = global_mean_sea_level
    # sort
    limited_altimetry_global_sea_level = {k: v for k, v in sorted(limited_altimetry_global_sea_level.items(),
                                                                  key=lambda item: item[0])}
    global_sea_level_altimetry_for_date = {}
    for date in weighted_altimetry_dataset.time.values:
        decimal_date = convert_to_decimal_date(date)
        data = np.array(weighted_altimetry_dataset.sel(time=date).weighted_sla.values)
        # calculate the mean sea level for where the value is not nan
        sum_global_altimetry_sea_level = 0
        no_of_values = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                if not np.isnan(data[i][j]):
                    sum_global_altimetry_sea_level += data[i][j]
                    no_of_values += 1
        global_mean_sea_level = sum_global_altimetry_sea_level / no_of_values
        global_sea_level_altimetry_for_date[decimal_date] = global_mean_sea_level
    return global_sea_level_altimetry_for_date, limited_altimetry_global_sea_level


def convert_to_decimal_date(date):
    timestamp = pd.Timestamp(date)
    year = timestamp.year
    day_of_year = timestamp.dayofyear
    decimal_date = year + (day_of_year - 1) / (365 if is_leap_year(year) else 366)
    return decimal_date


# Reconstruction: H_r = EOF(x,y)_m * alpha(t)_r ->
# find coefficient alph(t)_r by performing least square regression
# minimizing vector: v = P * EOF_m * alpha_r - H_o
# minimize S(alpha) = (KU^r alpha - H^o)^T M^-1 (KU^r alpha - H^o) + alpha^T Lambda alpha
# K = samping operator
# U^r = EOFs (the first m EOFs)
# H^o = the observed data from the tide gauge stations
# M error covariance matrix (M = R + KU'Lambda'U'^TK^T -> maybe we only want R)
# Lambda = diagonal matrix with the eigenvalues of the covariance matrix
# individual eigenvalues are related to the singular values by lambda_i = s_i^2 / n (where n is number of
# grid points)

def plot_eofs(eof_dataset, out_dir, number_of_eofs: int):
    """
    Plot the EOFs from the dataset
    :param eof_dataset:
    :param out_dir:
    :param number_of_eofs:
    :return:
    """
    # make images out of dataset and save them
    for i in range(number_of_eofs):
        data = eof_dataset.PAT.isel(pca=i)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Plot the DataArray
        data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', add_colorbar=True)
        # Add coastlines and gridlines
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        # save
        plt.savefig(os.path.join(out_dir, f"EOF_from_dataset_{i}.svg"), dpi=600)
        plt.close(fig)


def get_closest_grid_point(station_id, target_lat, target_lon, eof_dataset, outdir) -> (float, float):
    """
    Calculate the closest grid point to the given latitude and longitude for which there are values in the dataset
    :param eof_dataset:
    :param outdir:
    :param target_lon:
    :param target_lat:
    :param station_id:
    :return:
    """
    # station_lon is between 180 and -180, dataset longitude is between 0 and 360
    # station_lat is between 90 and -90, dataset latitude is between -90 and 90
    # check if the range of lat and lon is correct
    if target_lat < -90 or target_lat > 90:
        raise ValueError(f"Latitude {target_lat} is out of bounds.")
    if target_lon < -180 or target_lon > 180:
        raise ValueError(f"Longitude {target_lon} is out of bounds.")
    if eof_dataset.lat.values.min() < -90 or eof_dataset.lat.values.max() > 90:
        raise ValueError(f"Latitude in dataset is out of bounds.")
    if eof_dataset.lon.values.min() < -180 or eof_dataset.lon.values.max() > 180:
        # fix the longitude
        logger.warning("Longitude out of range - attempting to fix it")
        eof_dataset = eof_dataset.assign_coords(lon=(eof_dataset.lon + 180) % 360 - 180)
        eof_dataset = eof_dataset.sortby("lon")
    grid_resolution = 0.25
    lat_start = -89.875
    lon_start = 0.125
    lat_range = 5
    lon_range_at_equator = 5
    # Longitude bounds adjustment at the poles
    cos_lat = math.cos(math.radians(target_lat))
    lon_range = lon_range_at_equator * cos_lat

    min_lat = max(-90.0, target_lat - lat_range)
    max_lat = min(90.0, target_lat + lat_range)

    min_lon = target_lon - lon_range
    max_lon = target_lon + lon_range

    if min_lon < -180.0:
        min_lon += 360.0
    if max_lon > 180.0:
        max_lon -= 360.0
    lat_points = []
    lat = round((min_lat - lat_start) / grid_resolution) * grid_resolution + lat_start
    while lat <= max_lat:
        lat_points.append(round(lat, 6))  # rounding for precision
        lat += grid_resolution

    lon_points = []
    lon = min_lon
    while True:
        wrapped_lon = ((lon + 180.0) % 360.0) - 180.0
        wrapped_lon = round((wrapped_lon - lon_start) / grid_resolution) * grid_resolution + lon_start
        lon_points.append(round(wrapped_lon, 6))
        lon += grid_resolution
        if (min_lon < max_lon and lon > max_lon) or (min_lon > max_lon and lon > min_lon + 360.0 - (min_lon - max_lon)):
            break
    # Find the closest grid point
    results = []
    for lat in lat_points:
        for lon in lon_points:
            # calculate haversine distance,
            distance = haversine((target_lat, target_lon), (lat, lon))
            results.append((distance, lat, lon))
    results.sort(key=lambda x: x[0])
    best_distance = np.inf
    best_lat = None
    best_lon = None
    if "PAT" in eof_dataset.variables:
        for distance, lat, lon in results:
            # if dataset value is not none, this is the closest valid point
            if not np.isnan(eof_dataset.sel(lat=lat, lon=lon).PAT.values[0]):
                best_distance = distance
                best_lat = lat
                best_lon = lon
                break
    else:
        for distance, lat, lon in results:
            # if dataset value is not none, this is the closest valid point
            if not np.isnan(eof_dataset.sel(lat=lat, lon=lon).sla.values[0]):
                best_distance = distance
                best_lat = lat
                best_lon = lon
                break

    if best_distance > 300:
        with open(os.path.join(outdir, "distant_points.txt"), "a") as file:
            file.write(
                f"Closest point for station {station_id} at ({target_lat}, {target_lon}) is ({best_lat}, {best_lon}) "
                f"with distance {best_distance}.\n")
        # data = eof_dataset.PAT.isel(pca=0)
        # fig = plt.figure(figsize=(50, 25))
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # # Plot the DataArray
        # data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', add_colorbar=True)
        #
        # # plot all the points
        # # for distance, lat, lon in results:
        # #     ax.plot(lat, lon, 'o', transform=ccrs.PlateCarree(), label="Closest point", color="green")
        # # plot the closest point
        # if best_lat is not None:
        #     ax.plot(best_lon, best_lat, 'x', transform=ccrs.PlateCarree(), label="Closest point",
        #             color="red")
        # # plot the original point
        # ax.plot(target_lon, target_lat, 'o', transform=ccrs.PlateCarree(), label="Original point",
        #         color="magenta")
        # # Add coastlines and gridlines
        # ax.coastlines()
        # ax.gridlines(draw_labels=True)
        # # save
        # plt.savefig(os.path.join(outdir, f"closest_points{station_id}"), dpi=600)
        # plt.close(fig)
    return best_lat, best_lon
