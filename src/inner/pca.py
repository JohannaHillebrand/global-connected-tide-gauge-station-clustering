import os

import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from netCDF4 import Dataset
from sklearn.decomposition import PCA


def plot_pca_results(filepath: str, outdir: str):
    """
    plot the pattern dimension of this netdcf file on the globe
    :param filepath:
    :return:
    """
    with xr.open_dataset(filepath) as dataset:
        print(dataset)
        for i in range(10):
            data = dataset.PAT[:, :, i]
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.PlateCarree())
            data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', add_colorbar=True)
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            plt.savefig(os.path.join(outdir, f"EOF_{i}.png"), dpi=600)
            plt.close(fig)
    return


if __name__ == '__main__':
    # Define input and output paths
    inpath = '../../data/SEALEVEL_GLO_PHY_L4_MY_008_047'
    output_dir = '../../output/PCA'
    fileout = '../../output/PCA/PCA_SLA.nc'
    outdescr = 'Principal Component Analysis applied on SLA data.'

    # Load SLA data
    print('loading data..')

    dlist = [f for f in os.listdir(inpath) if f.endswith('.nc')]
    dlist.sort()

    # Load first file to get dimensions
    with xr.open_dataset(os.path.join(inpath, dlist[0])) as ds:
        # create mask with all values that are nan at any point in time set to False
        maskR = ds.sla.notnull().any(dim='time')
        print(f"mask shape: {maskR.shape}")
        print(f"false values in mask: {np.sum(~maskR)}")
        print(f"true values in mask: {np.sum(maskR)}")
        # maskR = ds.sla.isnull().any(dim='time')
        lat = ds.latitude.values
        lon = ds.longitude.values
        input_matrix = np.zeros((len(lat) * len(lon), len(dlist)))
        time = np.zeros(len(dlist))
        # plot this dataset
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ds.sla[0].plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', add_colorbar=True)
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        plt.savefig(os.path.join(output_dir, "first_dataset.png"), dpi=600)
        plt.close(fig)

    for j, file in enumerate(dlist):
        with xr.open_dataset(os.path.join(inpath, file)) as ds:
            input_matrix[:, j] = ds.sla.values.flatten()
            current_time = ds.time.values[0]
            pd_date = pd.to_datetime(current_time)
            year = pd_date.year
            month = pd_date.month
            time[j] = year + (month - 0.5) / 12

    # Filter data for years before 2023
    idx = time < 2023
    input_matrix = input_matrix[:, idx]
    print(input_matrix.shape)
    time = time[idx]

    print('read done')

    # Create mask and apply area weighting
    mask = ~np.isnan(input_matrix[:, 0])
    print(f"mask shape: {mask.shape}")
    print(f"false values in mask: {np.sum(~mask)}")
    print(f"true values in mask: {np.sum(mask)}")
    input_matrix[np.isnan(input_matrix)] = 0
    print(input_matrix.shape)

    ilat, ilon = np.meshgrid(lat, lon)
    ilat = ilat.flatten()

    area_weight = np.cos(np.radians(ilat))
    area_weight_sqrt = np.sqrt(area_weight)

    input_matrix = input_matrix * area_weight_sqrt[:, np.newaxis]

    # Calculate inverse area weight to undo the weighting after PCA
    reverse_area_weight = 1 / area_weight
    print(input_matrix[mask, :].T.shape)
    # Perform PCA
    print('calculating pca..')
    pca = PCA()
    score = pca.fit_transform(input_matrix[mask, :].T)
    coef = pca.components_.T
    latent = pca.explained_variance_

    print('done')
    # Post-process PCA results
    # Apply inverse area weighting
    reverse_area_weight_mask = reverse_area_weight[mask]
    coef = coef * reverse_area_weight_mask[:, np.newaxis]

    # Normalize patterns and scores
    nrmvec = np.max(np.abs(coef), axis=0)
    coef = coef / nrmvec
    score = score * nrmvec
    hv = np.full((input_matrix.shape[0], coef.shape[1]), np.nan)
    print(f"hv shape: {hv.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"coef shape: {coef.shape}")
    hv[mask, :] = coef
    coef = hv

    coef[~maskR.values.flatten(), :] = np.nan

    npca = coef.shape[1]
    coef = coef.reshape(len(lon), len(lat), npca)

    # Write PCA results to netCDF file
    print('writing pca data')

    with Dataset(fileout, 'w', format='NETCDF4') as nc:
        # Define dimensions
        nc.createDimension('lat', len(lat))
        nc.createDimension('lon', len(lon))
        nc.createDimension('time', len(time))
        nc.createDimension('pca', npca)

        # Create variables
        latitudes = nc.createVariable('lat', 'f8', ('lat',))
        longitudes = nc.createVariable('lon', 'f8', ('lon',))
        times = nc.createVariable('time', 'f8', ('time',))
        pc = nc.createVariable('PC', 'f8', ('time', 'pca'))
        pat = nc.createVariable('PAT', 'f8', ('lon', 'lat', 'pca'))
        eig = nc.createVariable('EIG', 'f8', ('pca',))

        # Add attributes
        latitudes.long_name = 'Latitude'
        latitudes.units = 'degrees_north'
        latitudes.actual_range = [np.min(lat), np.max(lat)]

        longitudes.long_name = 'Longitude'
        longitudes.units = 'degrees_east'
        longitudes.actual_range = [np.min(lon), np.max(lon)]

        times.long_name = 'Time_year'
        times.units = 'years'
        times.actual_range = [np.min(time), np.max(time)]

        eig.long_name = 'Eigenvalues (normalized wrt trace = % of variance explained)'

        pc.long_name = 'Principal_components'
        pc.units = 'm'

        pat.long_name = 'Orthogonal_patterns (normalized wrt maximum)'
        pat.units = '-'
        pat.missing_value = np.nan

        # Global attributes
        nc.convention = 'COARDS'
        nc.Description = outdescr

        # Write data
        latitudes[:] = lat
        longitudes[:] = lon
        times[:] = time
        eig[:] = latent / np.sum(latent)
        pc[:] = score
        pat[:] = coef
    plot_pca_results(fileout, output_dir)
    print('Done')
