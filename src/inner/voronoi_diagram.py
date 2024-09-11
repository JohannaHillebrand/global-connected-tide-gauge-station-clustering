import json
import os.path
import warnings

import geopandas
import matplotlib.pyplot as plt
import networkx
import plotly.express as px
import pyproj
import shapely
from loguru import logger
from srai.regionalizers import VoronoiRegionalizer

import src.inner.tide_gauge_station
from src.inner import tide_gauge_station, cluster_sections, timeseries_difference
from src.inner.plot import plot_voronoi
from src.inner.timeseries_difference import remove_percentages

# there is a runtime warning within the VoronoiRegionalizer, and i checked every possible origin, so we are ignoring
# it for now
warnings.filterwarnings("ignore", category=RuntimeWarning)


def assign_voronoi_areas_to_stations(voronoi_regions: geopandas.GeoDataFrame,
                                     stations: {str: tide_gauge_station.TideGaugeStation}):
    """
    Assign the voronoi areas to the stations
    :param voronoi_regions: 
    :param stations: 
    :return: 
    """
    stations_with_polygons = {}
    geod = pyproj.Geod(ellps='WGS84')
    for index, row in voronoi_regions.iterrows():
        # calculates the perimeter and area of a shapely polygon which are used in geopandas.GeoDataFrames
        poly_area, poly_perimeter = geod.geometry_area_perimeter(row.geometry)
        voronoi_regions.at[index, "area"] = poly_area
        stations[row["region_id"]].area = poly_area
        stations_with_polygons[row["region_id"]] = voronoi_regions.at[index, "geometry"]
    return stations_with_polygons


def calculate_voronoi_diagram(points_gdf: geopandas.GeoDataFrame):
    """
    Calculate the voronoi diagram
    :param points_gdf:
    :return:
    """
    # calculate voronoi regions
    regions = VoronoiRegionalizer(seeds=points_gdf).transform()
    # regions_without_land = regions.transform(gdf=ocean_polygon)
    return regions


def plot_on_globe(seeds_gdf: geopandas.GeoDataFrame, regions_gdf: geopandas.GeoDataFrame, output_path: str, lon: float,
                  lat: float, zoom: float = 1, marker_size: float = 5, title: str = None) -> None:
    """
    Plot the seeds and regions on a globe
    :param output_path:
    :param seeds_gdf:
    :param regions_gdf:
    :param lon:
    :param lat:
    :param zoom:
    :param marker_size:
    :param title:
    :return:
    """
    fig = px.choropleth(
        regions_gdf,
        geojson=regions_gdf.geometry,
        locations=regions_gdf.index,
        color=regions_gdf.index,
        color_continuous_scale=px.colors.qualitative.Bold,
    )
    fig2 = px.scatter_geo(seeds_gdf, lat=seeds_gdf.geometry.y, lon=seeds_gdf.geometry.x)
    fig.update_traces(marker={"opacity": 0.4}, selector=dict(type="choropleth"))
    fig.add_trace(fig2.data[0])
    fig.update_traces(
        marker_color="black", marker_size=marker_size, selector=dict(type="scattergeo")
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation_lon=lon,
        projection_rotation_lat=lat,
        showlakes=False,
        projection_scale=zoom,
    )
    fig.update_layout(
        height=800, width=800, margin={"r": 15, "t": 15, "l": 15, "b": 15}
    )
    if title:
        fig.update_layout(title=dict(text=title, automargin=True, x=0.5))
    plt.savefig(f"{output_path}/globe_voronoi.png", dpi=400)


def create_point_gdf(stations: {str: tide_gauge_station.TideGaugeStation}):
    """
    Create a geopandas dataframe with the stations
    :param stations:
    :return:
    """
    points_dict = {"id": [], "geometry": []}
    stations = remove_duplicate_stations(stations)
    for station in stations.values():
        points_dict["id"].append(station.id)
        points_dict["geometry"].append(shapely.Point(station.longitude, station.latitude))
    points_gdf = geopandas.GeoDataFrame(points_dict, crs="EPSG:4326", geometry=points_dict["geometry"],
                                        index=points_dict["id"])
    return points_gdf


def remove_duplicate_stations(stations: {str: tide_gauge_station.TideGaugeStation}):
    """
    Remove duplicate stations
    :param stations:
    :return:
    """
    duplicate_stations = {}
    for station in stations.values():
        duplicate_stations[station.id] = []
        is_duplicate = False
        if station.latitude >= 90 or station.latitude <= -90 or station.longitude >= 180 or station.longitude <= -180:
            print(f"stations coordinates are out of bounds: {station.id}, {station.latitude}, {station.longitude}")
        for station2_id in duplicate_stations:
            if station.id != station2_id:
                if station.longitude == stations[station2_id].longitude and station.latitude == stations[
                    station2_id].latitude:
                    duplicate_stations[station2_id].append(station.id)
                    is_duplicate = True
        if is_duplicate:
            duplicate_stations.pop(station.id)
    for station in duplicate_stations.keys():
        if duplicate_stations[station]:
            # remove duplicates
            for duplicate in duplicate_stations[station]:
                stations.pop(duplicate)
    return stations


def calculate_area(component, stations_with_polygons: {str: shapely.Polygon}):
    """
    Calculate the area of the components of the line graph based on the voronoi regions
    :param stations_with_polygons:
    :param component:
    :return:
    """
    counter = 0
    area = None
    for station_id in component:
        if counter == 0:
            area = stations_with_polygons[station_id]
        else:
            try:
                area = area.union(stations_with_polygons[station_id])
            except:
                print(f"Could not find station {station_id}")
        counter += 1

    return area


def voronoi_section_clustering(current_output_dir: str, land_path: str, metadata_path: str, ocean_path: str,
                               regions: geopandas.GeoDataFrame,
                               stations_for_time_step: {int: tide_gauge_station.TideGaugeStation},
                               wanted_number_centers: [int]):
    """
    Calculate the section clustering
    :param current_output_dir:
    :param land_path:
    :param metadata_path:
    :param ocean_path:
    :param regions:
    :param stations_for_time_step:
    :param wanted_number_centers:
    :return:
    """
    logger.info("Calculating section clustering")
    for number_of_centers in wanted_number_centers:
        # if the wanted number of centers is > than the number of stations, take all stations
        if number_of_centers >= len(stations_for_time_step.keys()):
            solution = {}
            for station in stations_for_time_step.keys():
                solution[station] = station
            with open(os.path.join(current_output_dir, f"solution{number_of_centers}.json"), "w") as outfile:
                json.dump(solution, outfile)
            continue
        regions_dict = cluster_sections.fill_region_dict(regions)
        cluster_sections.divide_and_cluster(current_output_dir, land_path, metadata_path, regions_dict,
                                            stations_for_time_step, number_of_centers)


def calculate_areas_per_graph(line_graph: networkx.Graph, stations_with_polygons: {str: shapely.Polygon}):
    """
    Calculate the areas per graph
    :param line_graph:
    :param stations_with_polygons:
    :return:
    """
    logger.info("Calculating areas per graph")
    regions = {}
    counter = 0
    for component in networkx.connected_components(line_graph):
        regions[str(counter)] = calculate_area(component, stations_with_polygons)
        counter += 1
    return regions


def calculate_time_series_differences(current_output_dir: str, metadata_path: str,
                                      stations_for_time_step: {int: tide_gauge_station.TideGaugeStation},
                                      time_step: str):
    """
    Calculate the time series differences between each pair of stations
    :param current_output_dir:
    :param metadata_path:
    :param stations_for_time_step:
    :param time_step:
    :return:
    """
    stations_for_time_step = src.inner.tide_gauge_station.detrend_and_mean_center_timeseries(stations_for_time_step)
    if not os.path.exists(os.path.join(current_output_dir, f"difference.txt")):
        logger.info(f"Calculating timeseries difference for time step: {time_step}")
        diffs_with_percentages = timeseries_difference.calculate_difference_between_all_pairs_of_stations(
            stations_for_time_step, metadata_path, True, False)
        diffs = remove_percentages(diffs_with_percentages)
        timeseries_difference.save_differences_to_file(diffs, current_output_dir, "difference.txt")
    else:
        diffs = timeseries_difference.read_differences_from_file(current_output_dir, "difference.txt")
    return diffs, stations_for_time_step


def determine_station_area(current_output_dir: str, end_year: float, land_path: str,
                           ocean_polygon: geopandas.GeoDataFrame, start_year: float,
                           stations: {str: tide_gauge_station.TideGaugeStation},
                           time_step: str):
    """
    Determine the area of each station
    :param current_output_dir:
    :param end_year:
    :param land_path:
    :param ocean_polygon:
    :param start_year:
    :param stations:
    :param time_step:
    :return:
    """
    stations_for_time_step = tide_gauge_station.filter_stations_for_time_step(stations, start_year, end_year)
    points_gdf = create_point_gdf(stations_for_time_step)
    if not os.path.exists(os.path.join(current_output_dir, "voronoi.shp")):
        voronoi_regions = calculate_voronoi_diagram(points_gdf)
        voronoi_regions.to_file(os.path.join(current_output_dir, "voronoi.shp"))
    voronoi_regions = geopandas.read_file(os.path.join(current_output_dir, "voronoi.shp"))
    stations_with_polygons = assign_voronoi_areas_to_stations(voronoi_regions, stations_for_time_step)
    logger.info(f"Number of stations in time step {time_step}: {len(stations_for_time_step)}")
    plot_voronoi(land_path, current_output_dir, points_gdf, voronoi_regions)
    return stations_for_time_step, stations_with_polygons
