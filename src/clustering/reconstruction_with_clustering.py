import json
import os

import xarray as xr
from loguru import logger
from matplotlib import pyplot as plt

from src.evaluation import evaluate_reconstruction
from src.inner import tide_gauge_station, reconstruction
from src.inner.reconstruction import calculate_global_mean_sea_level_for_altimetry_data, \
    calculate_and_compare_global_sea_level


def start(altimetry_data_path: str, out_dir: str, stations: {int: tide_gauge_station.TideGaugeStation},
          clustering_path: str, start_year: int, end_year: int, cluster_sizes: [int]):
    """
    Calculate the EOFs and use them to reconstruct the altimetry data.
    The altimetry data is in meter and the stations are in mm, thus the stations data is transformed to match.
    :param cluster_sizes:
    :param end_year:
    :param start_year:
    :param clustering_path:
    :param altimetry_data_path:
    :param out_dir:
    :param stations:
    :return:
    """
    number_of_eofs = 30
    # Read and weight the altimetry data (cos(lat) correction) - the altimetry data is a 720x1440 grid with 366 data
    # points each, latitude is from -90 to 90 and longitude is from 0 to 360
    logger.info("Read altimetry data")
    if not os.path.exists(os.path.join(out_dir, "weighted_altimetry.nc")):
        weighted_dataframe_altimetry = read_and_weight_altimetry_data(altimetry_data_path, out_dir)
    else:
        weighted_dataframe_altimetry = xr.open_dataset(
            os.path.join(out_dir, "weighted_altimetry.nc"))
        complete_dataframe_altimetry = xr.open_dataset(
            os.path.join(out_dir, "complete_altimetry.nc"))
        del complete_dataframe_altimetry
    # # print(complete_dataframe_altimetry.dims)
    # sla_data = weighted_dataframe_altimetry["weighted_sla"]
    # # make land mask where there are nan values
    # land_mask = sla_data.isnull()
    # # replace nan values with zeros to avoid errors in EOF analysis
    # sla_data_cleaned = sla_data.fillna(0)
    #
    # if sla_data_cleaned.isnull().any():
    #     print("Some NaNs still remain after masking and cleaning.")
    # else:
    #     print("No NaNs present, ready for EOF analysis.")
    #
    # model = xeofs.models.EOF()
    # model.fit(sla_data_cleaned, dim="time")
    # components = model.components()
    # # put nan values back in components
    #
    # # plot mode 1 on globe
    # fig = plt.figure(figsize=(10, 5))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # # use land mask to plot the model components only at the ocean
    # data = components.isel(mode=1)
    # data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', add_colorbar=True)
    # ax.coastlines()
    # ax.gridlines(draw_labels=True)
    # plt.savefig(os.path.join(out_dir, "EOF_1.png"))
    # plt.close(fig)

    # if the EOFs are not already calculated, calculate them
    # # TODO: why are the EOFs wrong and why do they not fill out the whole globe?
    # if not os.path.exists(os.path.join(out_dir, "EOFs.nc")):
    #     # turn the weighted data into a matrix where the there is a row for every stations time series and every
    #     column
    #     # is a time step
    #     input_matrix, index_dict = reconstruction.create_input_matrix(weighted_dataframe_altimetry)
    #     # perform PCA on the input matrix to reduce the dimensionality and extract the EOFs and PCs
    #     print("performing PCA")
    #     number_of_eofs = 10
    #     eof, pc = reconstruction.perform_pca(input_matrix, out_dir, number_of_eofs)
    #     # display the EOFs on the globe and save the plots
    #     print(f"EOFs: {eof.shape}")
    #     print(f"PCs: {pc.shape}")
    #     # remove weight from the EOFs
    #     eof = reconstruction.remove_weight_and_normalize_eof(eof, index_dict)
    #     eof_dataset = make_dataset_out_of_eofs_and_plot(eof, index_dict, weighted_dataframe_altimetry, out_dir, pc)
    # # read in the EOFs
    # eof_dataset = xr.open_dataset(os.path.join(out_dir, "EOFs.nc"))
    eof_dataset = xr.open_dataset("../output/PCA/PCA_AVISOGRIDDED_SLA.nc")
    # change lon from 0-360 to -180-180
    eof_dataset = eof_dataset.assign_coords(lon=(eof_dataset.lon + 180) % 360 - 180)
    eof_dataset = eof_dataset.sortby('lon')

    # weighted_dataframe_altimetry rename latitiude to lat and longitude to lon
    weighted_dataframe_altimetry = weighted_dataframe_altimetry.rename({'latitude': 'lat', 'longitude': 'lon'})

    # apply GIA (Glacial Isostatic Adjustment) to measured data (tide gauge data)
    stations = reconstruction.apply_gia_to_tide_gauge_data(stations)

    stations = tide_gauge_station.filter_timeseries_without_removing_stations(stations, start_year, end_year)
    stations = tide_gauge_station.mean_center_timeseries(stations)
    logger.info("Calculate global mean sea level for altimetry data")
    global_sea_level_altimetry_for_date, limited_altimetry_global_sea_level = (
        calculate_global_mean_sea_level_for_altimetry_data(
            number_of_eofs, eof_dataset, weighted_dataframe_altimetry))
    del weighted_dataframe_altimetry

    # for each date in time take all stations that are centers at that point in time and use the EOFs to reconstruct
    # the data with linear least squares regression (from numpy)
    all_rms = {}
    for cluster_size in cluster_sizes:
        logger.info(f"Reconstruction for cluster size: {cluster_size}")
        current_out_dir = os.path.join(out_dir, f"1992_2023/{cluster_size}/")
        os.makedirs(current_out_dir, exist_ok=True)
        file_name = f"solution{cluster_size}.json"

        rms = reconstruction_and_evaluation(cluster_size, clustering_path, end_year, eof_dataset, file_name,
                                            number_of_eofs, current_out_dir, start_year, stations,
                                            global_sea_level_altimetry_for_date, limited_altimetry_global_sea_level)
        all_rms[cluster_size] = rms
    with open(os.path.join(out_dir, "rms.json"), "w") as file:
        json.dump(all_rms, file)

    fig, ax = plt.subplots()
    ax.plot(cluster_sizes, all_rms.values())
    ax.set_xlabel("Number of cluster centers")
    ax.set_ylabel("RMS")
    ax.set_title("RMS for different number of cluster centers")
    plt.savefig(f"{out_dir}/rms.svg")
    plt.close()
    return


def reconstruction_and_evaluation(cluster_size, clustering_path, end_year, eof_dataset, file_name,
                                  number_of_eofs, out_dir, start_year, stations,
                                  global_sea_level_altimetry_for_date, limited_altimetry_global_sea_level):
    """
    Reconstruct the data and evaluate the reconstruction.
    :param limited_altimetry_global_sea_level:
    :param global_sea_level_altimetry_for_date:
    :param cluster_size:
    :param clustering_path:
    :param end_year:
    :param eof_dataset:
    :param file_name:
    :param number_of_eofs:
    :param out_dir:
    :param start_year:
    :param stations:
    :return:
    """
    reconstructed_dataset = reconstruction.reconstruct_data(eof_dataset, stations, clustering_path,
                                                            file_name, out_dir,
                                                            number_of_eofs, start_year, end_year)
    # check the reconstructed data against the tide gauge stations that were not used in the reconstruction
    time_steps = [(i, i + 10) for i in range(start_year, end_year, 10)]
    evaluate_reconstruction.start(reconstructed_dataset, stations, cluster_size, time_steps, clustering_path,
                                  out_dir)
    # validate the results with the global mean sea level for time >= 1993
    rms = 0
    rms = calculate_and_compare_global_sea_level(out_dir, reconstructed_dataset, global_sea_level_altimetry_for_date,
                                                 limited_altimetry_global_sea_level)
    del reconstructed_dataset
    return rms


def read_and_weight_altimetry_data(altimetry_data_path, out_dir):
    """
    Read and weight the altimetry data.
    :param altimetry_data_path:
    :param out_dir:
    :return:
    """
    complete_dataframe_altimetry = reconstruction.read_altimetry_data(altimetry_data_path)
    # recenter dataset to -180 to 180 longitude
    complete_dataframe_altimetry = complete_dataframe_altimetry.assign_coords(
        longitude=(complete_dataframe_altimetry.longitude + 180) % 360 - 180)
    complete_dataframe_altimetry = complete_dataframe_altimetry.sortby('longitude')
    logger.info("weight altimetry data")
    weighted_dataframe_altimetry = reconstruction.weight_altimetry_data(complete_dataframe_altimetry)
    logger.info("Reading altimetry data done")
    # safe the weighted altimetry data
    weighted_dataframe_altimetry.to_netcdf(
        os.path.join(out_dir, "weighted_altimetry.nc"))
    complete_dataframe_altimetry.to_netcdf(
        os.path.join(out_dir, "complete_altimetry.nc"))
    return weighted_dataframe_altimetry
