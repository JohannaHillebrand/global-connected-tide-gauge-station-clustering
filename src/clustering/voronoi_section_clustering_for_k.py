import os
import time

import geopandas
from loguru import logger

from src.inner import tide_gauge_station, sea_level_line_graph, plot
from src.inner.cluster_sections import fill_region_dict
from src.inner.voronoi_diagram import determine_station_area, calculate_time_series_differences, \
    calculate_areas_per_graph, voronoi_section_clustering


def start(station_path: str, output_dir: str, land_path: str, ocean_path: str, time_steps: [(int, int)],
          wanted_number_centers: [int]):
    """
    Start the voronoi diagram calculations
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metadata_path = os.path.join(output_dir, "metadata.txt")
    stations = tide_gauge_station.read_and_create_stations(station_path, metadata_path)

    # read ocean polygon
    ocean_polygon = geopandas.read_file(ocean_path)
    time1 = time.time()
    for time_step in time_steps:
        start_year = time_step[0]
        end_year = time_step[1]
        current_output_dir = os.path.join(output_dir, f"{start_year}_{end_year}")
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        # filter stations for the time step
        logger.info("Calculating voronoi diagrams")
        stations_for_time_step, stations_with_polygons = determine_station_area(current_output_dir, end_year, land_path,
                                                                                ocean_polygon, start_year, stations,
                                                                                time_step)
        # calculate timeseries difference
        diffs, stations_for_time_step = calculate_time_series_differences(current_output_dir, metadata_path,
                                                                          stations_for_time_step, time_step)
        # calculate line graph
        logger.info("Calculating line graph")
        line_graph = sea_level_line_graph.construct_line_graph_for_current_timestep(metadata_path, diffs,
                                                                                    list(
                                                                                        stations_for_time_step.values()))
        # calculate area per graph based on voronoi diagram
        regions = calculate_areas_per_graph(line_graph, stations_with_polygons)
        # plot all stations
        plot.plot_existing_stations(geopandas.read_file(land_path), os.path.join(current_output_dir, "all_stations"),
                                    stations, tide_gauge_station.gdf_from_stations(stations))
        plot_line_graph_and_regions(current_output_dir, land_path, line_graph, regions,
                                    stations_for_time_step)

        # calculate section clustering
        voronoi_section_clustering(current_output_dir, land_path, metadata_path, ocean_path, regions,
                                   stations_for_time_step, wanted_number_centers)
    time2 = time.time()
    logger.info(f"Entire calculation took {time2 - time1} s")
    with open(metadata_path, "a") as file:
        file.write(f"Entire calculation took {time2 - time1} s")
    return


def plot_line_graph_and_regions(current_output_dir, land_path, line_graph, regions, stations_for_time_step):
    # plot the line graph and the regions on the globe
    stations_gdf = tide_gauge_station.gdf_from_stations(stations_for_time_step)
    line_graph_gdf = sea_level_line_graph.gdf_from_graph(line_graph)
    regions_dict = fill_region_dict(regions)
    regions_gdf = geopandas.GeoDataFrame(regions_dict, crs="EPSG:4326")
    land_gdf = geopandas.read_file(land_path)
    regions_gdf = geopandas.overlay(regions_gdf, land_gdf, how="difference")
    plot.plot_line_graph_and_regions(current_output_dir, line_graph_gdf, regions_gdf, land_path, stations_gdf)
