import json
import multiprocessing
import os.path
import time

import geopandas
import pyproj
import shapely
from loguru import logger

import src.inner.tide_gauge_station
from src.inner import tide_gauge_station, line_graph_clustering, sea_level_line_graph, timeseries_difference, \
    voronoi_diagram, plot


def calculate_number_of_stations_per_region(regions_gdf: geopandas.GeoDataFrame,
                                            stations: {int: tide_gauge_station.TideGaugeStation}):
    """
    Calculate the density of stations in each region
    :param regions_gdf:
    :param stations:
    :return:
    """
    stations = stations.copy()
    time1 = time.time()
    station_dict = {"id": [], "color": [], "geometry": []}
    total_number_of_stations = 0
    for index, row in regions_gdf.iterrows():
        associated_stations = []
        region = row.geometry
        tuples = []
        for station in stations.values():
            tuples.append((station, region))
        with multiprocessing.Pool() as pool:
            associated_stations = pool.map(station_assignment, tuples)
        associated_stations = [x for x in associated_stations if x is not None]
        regions_gdf.at[index, "associated_stations"] = associated_stations
        regions_gdf.at[index, "number_of_stations"] = len(associated_stations)
        for station_id in associated_stations:
            station_dict["id"].append(station_id)
            station_dict["color"].append(row["color"])
            station_dict["geometry"].append(shapely.Point(stations[station_id].longitude,
                                                          stations[station_id].latitude))
            total_number_of_stations += 1
    for index, row in regions_gdf.iterrows():
        for station_id in row["associated_stations"]:
            stations.pop(station_id)
    for station in stations.values():
        print(f"Station {station.id} {station.latitude} {station.longitude} is not in any region")
    # if len(stations.keys()) == 0:
    #     print("All stations are in a region")
    # print(f"Total number of stations: {total_number_of_stations}")
    stations_gdf = geopandas.GeoDataFrame(station_dict, crs="EPSG:4326")
    time2 = time.time()
    # print(f"Time taken for calculating number of stations per region: {time2 - time1}")
    return regions_gdf, stations_gdf


def station_assignment(current_tuple: (tide_gauge_station.TideGaugeStation, shapely.Polygon)):
    """
    Assign a station to a region
    :param current_tuple:
    :return:
    """
    station, region = current_tuple
    if region.contains(shapely.Point(station.longitude, station.latitude)):
        return station.id
    return None


def divide_earth(land_gdf: geopandas.GeoDataFrame, output_path: str, stations_gdf: geopandas.GeoDataFrame,
                 regions_gdf: geopandas.GeoDataFrame,
                 stations: {int: tide_gauge_station.TideGaugeStation}):
    """
    Divide the earth into regions
    :param stations:
    :param regions_gdf:
    :param land_gdf:
    :param output_path:
    :param stations_gdf:
    :param ocean_gdf:
    :return:
    """
    # calculate the number of stations in each region
    # do this before dividing the earth, because the stations are too close to the coast and might end up
    #  not being in the ocean
    regions_gdf, stations_for_timestep_gdf = calculate_number_of_stations_per_region(regions_gdf,
                                                                                     stations)
    plot.plot_regions(land_gdf, output_path, stations_for_timestep_gdf, regions_gdf,
                      "regions_with_stations")
    # clip individual sections such that the land is not included
    regions_gdf = geopandas.overlay(regions_gdf, land_gdf, how="difference")
    # plot regions on map
    plot.plot_regions(land_gdf, output_path, stations_gdf, regions_gdf, "regions")
    # calculates the perimeter and area (in meters^2) of a shapely polygon which are used in geopandas.GeoDataFrames
    geod = pyproj.Geod(ellps='WGS84')
    for index, row in regions_gdf.iterrows():
        poly_area, poly_perimeter = geod.geometry_area_perimeter(row.geometry)
        regions_gdf.at[index, "area"] = abs(poly_area)
    return regions_gdf, stations_for_timestep_gdf


def calculate_density(wanted_number_of_stations: int, regions_gdf: geopandas.GeoDataFrame):
    """
    Calculate the density of stations and determine how many stations should be selected from each region
    :param wanted_number_of_stations:
    :param regions_gdf:
    :return:
    """
    change = True
    while change:
        ocean_area = 0  # in meter^2
        for index, row in regions_gdf.iterrows():
            if not row["all_selected"]:
                ocean_area += row["area"]
        if ocean_area == 0:
            break
        wanted_density = wanted_number_of_stations / ocean_area
        # calculate the amount of stations that should be selected from each region, if one station has less than the
        # wanted stations, redistribute the stations for the other regions
        remaining_stations = 0
        for index, row in regions_gdf.iterrows():
            if not row["all_selected"]:
                if row["number_of_stations"] == 1:
                    regions_gdf.at[index, "wanted_number_of_centers"] = 1
                    regions_gdf.at[index, "all_selected"] = True
                regions_gdf.at[index, "wanted_number_of_centers"] += round(row["area"] * wanted_density)
                if regions_gdf.at[index, "wanted_number_of_centers"] > row["number_of_stations"]:
                    difference = regions_gdf.at[index, "wanted_number_of_centers"] - row["number_of_stations"]
                    remaining_stations += difference
                    regions_gdf.at[index, "wanted_number_of_centers"] = row["number_of_stations"]
                    regions_gdf.at[index, "all_selected"] = True
        wanted_number_of_stations = remaining_stations
        if remaining_stations == 0:
            change = False

    # print(regions_gdf[["wanted_number_of_centers", "number_of_stations"]])
    return regions_gdf


def calculate_solution(land_path: str, output_dir: str, regions_gdf: geopandas.GeoDataFrame,
                       stations: {int: tide_gauge_station.TideGaugeStation}, stations_gdf: geopandas.GeoDataFrame,
                       metadata_path: str, wanted_number_of_centers: int):
    """
    Calculate the solution for each region, if the ""all_selected"" value is True, the region is already solved and
    every station in this area needs to be selected
    :param wanted_number_of_centers:
    :param metadata_path:
    :param land_path:
    :param output_dir:
    :param regions_gdf:
    :param stations:
    :param stations_gdf:
    :return:
    """
    overall_max_radius = 0
    other_output_path = os.path.join(output_dir, "other_output.txt")
    final_solution = {}
    for index, row in regions_gdf.iterrows():
        if row["all_selected"]:
            for station_id in row["associated_stations"]:
                final_solution[station_id] = [station_id]
        if row["wanted_number_of_centers"] == 0:
            continue
        else:
            output_dir_for_region = os.path.join(output_dir, f"region_{index}")
            if not os.path.exists(output_dir_for_region):
                os.makedirs(output_dir_for_region)
            current_stations = {station_id: stations[station_id] for station_id in row["associated_stations"]}
            # calculate sea level differences
            current_stations = src.inner.tide_gauge_station.detrend_and_mean_center_timeseries(current_stations)
            diffs_with_percentages = timeseries_difference.calculate_difference_between_all_pairs_of_stations(
                current_stations, other_output_path, True, False)
            diffs = {}
            for key in diffs_with_percentages.keys():
                diffs[key] = {}
                for second_key in diffs_with_percentages[key].keys():
                    diffs[key][second_key] = diffs_with_percentages[key][second_key][1]
            # calculate line graph
            line_graph = sea_level_line_graph.construct_line_graph_for_current_timestep(other_output_path, diffs,
                                                                                        list(current_stations.values()))
            plot.plot_line_graph(geopandas.read_file(land_path), output_dir_for_region, stations_gdf,
                                 sea_level_line_graph.gdf_from_graph(line_graph), f"line_graph{index}")
            # calculate clustering with k = wanted number of centers

            k = row["wanted_number_of_centers"]
            if k == 0:
                logger.info(f"k is zero for region {index}")
                continue
            result = line_graph_clustering.cluster_for_k(k,
                                                         line_graph, diffs,
                                                         None, 0)
            if result is not None:
                clustering, max_radius, elapsed_time, number_iterations = result
                if max_radius > overall_max_radius:
                    overall_max_radius = max_radius
                final_solution = {**final_solution, **clustering}
            else:
                print(f"Could not find a solution for region {index}, wanted solution of size {k}")
    print(
        f"Final solution size: {len(final_solution)} for year {output_dir.split('/')[-1]} with k = "
        f"{wanted_number_of_centers}")
    # plot final solution:
    center_dict = {"id": [], "color": [], "geometry": []}
    for station_id in final_solution.keys():
        center_dict["id"].append(station_id)
        center_dict["color"].append("green")
        center_dict["geometry"].append(shapely.Point(stations[station_id].longitude, stations[station_id].latitude))
    plot.plot_existing_stations(geopandas.read_file(land_path),
                                os.path.join(output_dir, f"clustered_solution{wanted_number_of_centers}"),
                                center_dict, geopandas.GeoDataFrame(center_dict, crs="EPSG:4326"))
    with open(os.path.join(output_dir, f"solution{wanted_number_of_centers}.json"), "w") as file:
        json.dump(final_solution, file)
    with open(os.path.join(output_dir, f"metadata.txt"), "a") as file:
        file.write(f"Wanted number of centers: {wanted_number_of_centers}\n")
        file.write(f"Number of centers: {len(final_solution)}\n")
        file.write(f"Max radius: {overall_max_radius}\n")


def fill_region_dict(regions: {str: shapely.Polygon}):
    """
    Fill the region dictionary with the regions and their properties
    :param regions:
    :return:
    """
    regions_dict = {"name": [], "color": [], "area": [], "number_of_stations": [], "wanted_number_of_centers": [],
                    "all_selected": [], "associated_stations": [], "geometry": []}
    colors = plot.random_color_generator(len(regions.keys()) + 1)
    counter = 0
    for region in regions.keys():
        regions_dict["name"].append(region)
        regions_dict["color"].append(colors[counter])
        regions_dict["area"].append(0.0)
        regions_dict["number_of_stations"].append(0)
        regions_dict["wanted_number_of_centers"].append(0)
        regions_dict["all_selected"].append(False)
        regions_dict["associated_stations"].append([])
        regions_dict["geometry"].append(regions[region])
        counter += 1
    return regions_dict


def divide_and_cluster(current_output_dir: str, land_path: str, metadata_path: str,
                       regions_dict: {str: shapely.Polygon},
                       stations_for_time_step: {int: tide_gauge_station.TideGaugeStation},
                       wanted_number_of_centers: int):
    """
    Divide the earth into regions and cluster the stations in each region
    :param current_output_dir:
    :param land_path:
    :param metadata_path:
    :param ocean_path:
    :param regions_dict:
    :param stations_for_time_step:
    :param wanted_number_of_centers:
    :return:
    """
    regions_gdf = geopandas.GeoDataFrame(regions_dict, crs="EPSG:4326")
    regions_gdf, stations_for_time_step_gdf = divide_earth(geopandas.read_file(land_path), current_output_dir,
                                                           voronoi_diagram.create_point_gdf(stations_for_time_step),
                                                           regions_gdf, stations_for_time_step)
    # how many stations should be selected from each region, such that the density is the same in each region,
    # by density we mean the number of stations per area
    regions_gdf = calculate_density(wanted_number_of_centers, regions_gdf)
    calculate_solution(land_path, current_output_dir, regions_gdf, stations_for_time_step,
                       stations_for_time_step_gdf, metadata_path, wanted_number_of_centers)
