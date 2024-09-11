import os

import shapely
from loguru import logger

import src.start_clustering.section_clustering_for_k
import src.start_clustering.voronoi_section_clustering_for_k
from src.inner import tide_gauge_station
from src.start_clustering import connected_clustering_for_k, connected_clustering_for_radius, \
    cluster_coastline_graph_for_radius, reconstruction_with_clustering, voronoi_section_clustering_for_k

# TODO: add automatic download of the most recent data if not otherwise specified by the user
# TODO: check out deployment options

if __name__ == "__main__":
    # ----------------------------------------------------9
    # are clusterings per time steps or for the complete timespan used
    step_length = 10
    start_year = 1992
    end_year = 2024
    out_dir = "../output/PCA/voronoi/30_eofs/"
    # The tide gauge stations can be found here
    station_list_file_path = "../data/rlr_monthly/filelist.txt"
    land_path = "../data/ne_10m_land/ne_10m_land.shp"  # used as background for plotting
    ocean_path = "../data/ocean_polygon/ne_10m_ocean.shp"

    # ----------------------------------------------------
    # Setup
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    station_info_path = os.path.join(out_dir, "station_info.txt")
    stations = tide_gauge_station.read_and_create_stations(station_list_file_path, station_info_path)
    time_steps = [(i, i + step_length) for i in range(start_year, end_year, step_length)]
    for time_step in time_steps:
        if time_step[1] > end_year:
            time_step = (time_step[0], end_year)

    # ----------------------------------------------------
    # calculate the voronoi diagram and use graphs for the section clustering
    calculate_voronoi_diagram = False
    if calculate_voronoi_diagram:
        # wanted_number_centers_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        wanted_number_centers_list = [1600]
        voronoi_section_clustering_for_k.start(station_list_file_path, out_dir, land_path, ocean_path, time_steps,
                                               wanted_number_centers_list)

    # ----------------------------------------------------

    # if cluster area is set, the stations are filtered
    cluster_area = False
    if cluster_area:
        regions = {"west_north_america": shapely.Polygon(((-180, 90), (-100, 90), (-100, 0), (-180, 0), (-180, 90))),
                   "east_north_america": shapely.Polygon(((-100, 90), (-20, 90), (-20, 0), (-100, 0), (-100, 90))),
                   "south_america": shapely.Polygon(((-180, 0), (-20, 0), (-20, -90), (-180, -90), (-180, 0))),
                   "south_africa": shapely.Polygon(((-20, 0), (70, 0), (70, -90), (-20, -90), (-20, 0))),
                   "australia": shapely.Polygon(((70, 0), (180, 0), (180, -90), (70, -90), (70, 0))),
                   "japan": shapely.Polygon(((70, 0), (180, 0), (180, 90), (70, 90), (70, 0))),
                   "europe": shapely.Polygon(((-20, 0), (70, 0), (70, 90), (-20, 90), (-20, 0)))}
        wanted_number_centers_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        for wanted_number_centers in wanted_number_centers_list:
            logger.info(f"Section-clustering for number of centers: {wanted_number_centers}")
            src.start_clustering.section_clustering_for_k.start(regions, station_list_file_path, time_steps, land_path,
                                                                out_dir,
                                                                wanted_number_centers)

    # ----------------------------------------------------
    # Connected clustering for k
    connected_clustering_k = False
    if connected_clustering_k:
        list_of_k = [600, 650, 700]
        mean_center_and_detrend = True
        rms = True
        mae = False
        reduce_graph_per_time_step = False
        # How much overlap between a cluster center and its associated stations is wanted (in percent)
        percentage = 90
        connected_clustering_for_k.start(list_of_k, time_steps, stations, out_dir,
                                         mean_center_and_detrend, rms, mae, percentage, reduce_graph_per_time_step,
                                         land_path)
    # ----------------------------------------------------
    # Connected clustering for radius
    connected_clustering_radius = False
    if connected_clustering_radius:
        # for the PSMSL dataset, the sea level is given in mm
        list_of_radii = [0.0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        # list_of_radii = [0.0]
        # for the ORAS5 dataset, the sea level is given in m
        # list_of_radii = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
        #                  0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
        mean_center_and_detrend = True
        rms = True
        mae = False
        percentage = 90
        reduce_graph_per_time_step = False
        connected_clustering_for_radius.start(list_of_radii, time_steps, stations, out_dir,
                                              mean_center_and_detrend, rms, mae, percentage, reduce_graph_per_time_step,
                                              land_path)

    cluster_coastline_line_graph = False
    if cluster_coastline_line_graph:
        # list_of_radii = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        # list_of_radii = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
        #                  0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
        mean_center_and_detrend = False
        rms = True
        mae = False
        percentage = None
        coastline_line_graph_path = "../data/coastline_line_graph/oras5/line_graphs_edgelist.csv"
        node_path = "../data/coastline_line_graph/oras5/node_data.txt"
        cluster_coastline_graph_for_radius.start(list_of_radii, coastline_line_graph_path, node_path, stations,
                                                 out_dir, mean_center_and_detrend, rms, mae)

    reconstruction = True
    if reconstruction:
        altimetry_data_path = "../data/SEALEVEL_GLO_PHY_L4_MY_008_047/"
        clustering_path = "../output/Voronoi/"

        all_rms = {}
        start_year = 1992
        end_year = 2023
        cluster_sizes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        # cluster_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

        rms = reconstruction_with_clustering.start(altimetry_data_path, out_dir, stations, clustering_path, start_year,
                                                   end_year, cluster_sizes)
