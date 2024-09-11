import colorsys
import json
import os
import random

import geopandas
import matplotlib
import shapely
from geopandas import GeoDataFrame
from geopandas import read_file
from matplotlib import pyplot as plt, dates as mdates
from shapely import Point

from src.inner import tide_gauge_station

matplotlib.use("cairo")
plt.rcParams.update({'font.size': 20})


def plot_timelines(time_series_to_plot: [(dict, str, str)], name: str, output_dir: str):
    """
    Plot the global mean sea level, the mean sea level of all tide gauge stations and the clustered solution
    :param output_dir:
    :param name:
    :param time_series_to_plot:
    :return:
    """
    # plot complete clustered solution
    try:
        fig, ax = plt.subplots(figsize=(20, 12))
        counter = 0
        for time_series, label, color in time_series_to_plot:
            counter += 1
            ax.plot(*zip(*sorted(time_series.items())), zorder=counter, color=color, label=label, linewidth=4)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        plt.xlabel("time")
        plt.ylabel("sea level")
        plt.savefig(os.path.join(output_dir, f"{name}.svg"))
        plt.close()
    except Exception as e:
        print(time_series_to_plot)


def plot_rmse_graph(rmse_to_plot: [()], name: str, output_dir: str, x_label: str):
    """
    Plot the RMSE graph
    :param x_label:
    :param name:
    :param rmse_to_plot:
    :param output_dir:
    :return:
    """
    # make line graph plot of all RMSE values
    fig, ax = plt.subplots(figsize=(20, 12))
    counter = 0
    for rmse, label, color in rmse_to_plot:
        counter += 1
        ax.plot(*zip(*sorted(rmse.items())), zorder=counter, color=color, label=label, linewidth=4)
    plt.xlabel(x_label, fontsize=30)
    plt.ylabel("RMS", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    box = ax.get_position()
    if x_label == "Number of centers":
        ax.xaxis.set_inverted(True)
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    ax.legend(fontsize=30, loc="upper left")
    # plt.yticks(range(14, 32, 2))
    plt.savefig(os.path.join(output_dir, f"{name}.svg"))
    plt.close()


def plot_points_on_world(points: [shapely.Point], name: str, land_path, output_dir: str):
    """
    Plot the points on a world map
    :param points:
    :param name:
    :param output_dir:
    :return:
    """
    # read in land data
    land_gdf = geopandas.read_file(land_path)
    land_gdf = land_gdf.explode("geometry", ignore_index=True)
    # plot points on world map
    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    for point in points:
        ax.plot(point.x, point.y, color="green", marker="o", markersize=4, zorder=1)
    plt.savefig(os.path.join(output_dir, f"{name}.svg"))
    plt.close()


def plot_voronoi(land_path, output_dir, points_gdf, regions):
    world = geopandas.read_file(land_path)
    world = world.to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(20, 12))
    world.plot(ax=ax, color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    regions.reset_index().plot(ax=ax, alpha=0.4, column="region_id", edgecolor="black", linewidth=0.5)
    points_gdf.plot(ax=ax, alpha=0.6, color="black", markersize=1)
    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"voronoi.svg"))
    plt.close()


def plot_line_graph(land_gdf, output_path, present_stations_gdf, result_gdf, name: str):
    if result_gdf.empty:
        return
    # plot results
    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    result_gdf.plot(ax=ax, color=result_gdf["color"], zorder=4, linewidth=4)
    if not present_stations_gdf.empty:
        present_stations_gdf.plot(ax=ax, color="green", markersize=0.1, zorder=3)
    plt.savefig(os.path.join(output_path, f"{name}.svg"))
    plt.close()


def plot_existing_stations(land_gdf, plot_path, stations_for_plotting, stations_for_plotting_gdf):
    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    stations_for_plotting_gdf.plot(ax=ax, color="green", markersize=5, zorder=2)
    plt.savefig(f"{plot_path}.svg")
    plt.close()


def plot_difference_histogram(differences: dict, out_path: str):
    """
    Plot a histogram showing the distribution of the difference values between all pairs of stations
    :param differences:
    :param out_path:
    :return:
    """
    all_differences = {}
    for key in differences.keys():
        for sub_key in differences[key].keys():
            if differences[key][sub_key][1] != float("inf"):
                if differences[key][sub_key][1] <= 200:
                    if round(differences[key][sub_key][1]) in all_differences.keys():
                        all_differences[round(differences[key][sub_key][1])] += 1
                    else:
                        all_differences[round(differences[key][sub_key][1])] = 1

    # print(all_differences)
    tuples = [(key, value) for key, value in all_differences.items()]
    tuples = sorted(tuples, key=lambda x: x[0])

    plt.plot([x[0] for x in tuples], [x[1] for x in tuples])
    plt.x_ticks = range(0, 150, 10)
    plt.xlabel("RMS difference in sea level")
    plt.savefig(os.path.join(out_path, "difference_histogram.svg"))
    plt.close()
    return


def plot_all_stations(present_stations, out_path):
    """
    Plot stations that are currently present in the list of stations
    :param present_stations:
    :param out_path:
    :return:
    """
    landmass_file_path = "../data/ne_10m_land/ne_10m_land.shp"
    land_gdf = read_file(landmass_file_path)
    land_gdf = land_gdf.explode("geometry", ignore_index=True)
    geometry = []
    station_id = []
    color = []
    for station in present_stations.values():
        geometry.append(Point(station.longitude, station.latitude))
        station_id.append(station.id)
        color.append("green")
    stations_for_plotting = {"id": station_id, "color": color, "geometry": geometry}
    stations_for_plotting_gdf = GeoDataFrame.from_dict(stations_for_plotting, orient="columns")

    plot_path = os.path.join(out_path, "existing_stations")
    plot_existing_stations(land_gdf, plot_path, stations_for_plotting, stations_for_plotting_gdf)
    return


def plot_current_year(current_stations, geometry, old_id, land_gdf, year):
    """
    Plot all stations that are present in a given year
    :param old_id:
    :param current_stations:
    :param geometry:
    :param land_gdf:
    :param year:
    :return:
    """
    station_id = []
    color = []
    working_stations = current_stations.copy()
    for station in working_stations.values():
        dates_to_remove = []
        for date in station.timeseries.keys():
            if station.timeseries[date] == -99999:
                dates_to_remove.extend([date])
        for date in dates_to_remove:
            station.timeseries.pop(date)
            station.timeseries_detrended_normalized.pop(date)

    for station in working_stations.values():
        for date in station.timeseries.keys():
            if int(date) == year:
                station_id.append(station.id)
                if station.id in old_id:
                    color.append("green")
                else:
                    color.append("orange")
                geometry.append(Point(station.longitude, station.latitude))
                break

    present_stations = {"id": station_id, "color": color, "geometry": geometry}
    present_stations_gdf = GeoDataFrame.from_dict(present_stations, orient="columns")
    # plot everything without the graph
    plot_path = "../output/plot/yearly/img_" + str(year) + ".svg"
    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    present_stations_gdf.plot(ax=ax, color=color, markersize=5, zorder=2)
    plt.xlabel(f"{year}", fontsize=16)
    plt.savefig(plot_path)
    # plt.show()
    plt.close()
    return station_id


def plot_regions(land_gdf: geopandas.GeoDataFrame, output_path: str, stations_gdf: geopandas.GeoDataFrame,
                 regions_gdf: geopandas.GeoDataFrame, name: str):
    """
    Plot the regions on a map
    :param name:
    :param land_gdf:
    :param output_path:
    :param stations_gdf:
    :param regions_gdf:
    :param param:
    :return:
    """
    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    regions_gdf.plot(ax=ax, color=regions_gdf["color"], zorder=4, linewidth=4, alpha=0.1)
    regions_gdf.boundary.plot(ax=ax, color=regions_gdf["color"], zorder=5, linewidth=0.5)
    if "color" in stations_gdf.columns:
        stations_gdf.plot(ax=ax, color=stations_gdf["color"], markersize=0.1, zorder=3)
    else:
        stations_gdf.plot(ax=ax, color="green", markersize=0.1, zorder=3)
    plt.xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks([-90, -45, 0, 45, 90])
    plt.savefig(os.path.join(output_path, f"{name}.svg"))
    plt.close()
    return


def random_color_generator(num_colors: int):
    """
    Generates a list of random colors
    :param num_colors:
    :return:
    """
    colors = []

    for i in range(num_colors - 1):
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
        colors.append('#%02x%02x%02x' % (r, g, b))
        # colors.append(random.choice(list(mcolors.CSS4_COLORS.keys())))
    return colors


def plot_timelines_and_save(stations_to_plot, out_dir, land_directory: str):
    """
    Plots the timelines of the selected stations
    :param land_directory:
    :param stations_to_plot:
    :param out_dir:
    :return:
    """
    land_gdf = geopandas.read_file(land_directory)
    land_gdf = land_gdf.explode("geometry", ignore_index=True)
    station_dict = {"id": [], "geometry": []}
    for current_station in stations_to_plot:
        with open(os.path.join(out_dir, "metadata.txt"), "w") as file:
            file.write(
                f"{current_station.id}; {current_station.name}; {current_station.latitude}; "
                f"{current_station.longitude}\n")
        station_dict["id"].append(current_station.id)
        station_dict["geometry"].append(shapely.geometry.Point(current_station.longitude, current_station.latitude))
    station_gdf = geopandas.GeoDataFrame(station_dict)
    ax = land_gdf.plot(color="burlywood", figsize=(40, 24), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    station_gdf.plot(ax=ax, color="red", marker=",", markersize=0.01, zorder=1)
    for i, txt in enumerate(station_gdf["id"]):
        ax.annotate(txt, (station_gdf["geometry"].iloc[i].x, station_gdf["geometry"].iloc[i].y), fontsize=0.1)
    plt.savefig(os.path.join(out_dir, "selected_stations.svg"))
    plt.close()
    # plot timelines
    fig, ax = plt.subplots(figsize=(40, 24))
    colors = random_color_generator(len(stations_to_plot) + 100)
    # colors = ["azure", "yellow", "blue", "red", "green", "purple", "teal", "pink",
    #           "lightblue", "darkblue", "brightpink",
    #           "violet", "lightgreen"]
    counter = 0
    for current_station in stations_to_plot:
        dates_to_remove = []
        for date in current_station.timeseries:
            if current_station.timeseries[date] == -99999 or date < 2010 or date > 2020:
                dates_to_remove.append(date)
        for date in dates_to_remove:
            current_station.timeseries.pop(date)
            current_station.timeseries_detrended_normalized.pop(date)
        ax.plot(*zip(*sorted(current_station.timeseries.items())), label=current_station.id, color=colors[counter],
                linewidth=4)
        counter += 1
    plt.legend()
    plt.savefig(os.path.join(out_dir, "selected_stations_timeline_2000_to_present.svg"))
    plt.close()
    return


def plot_groups_of_stations():
    # read in stations
    output_path = "../output/analyze_clustering/golf_bengalen/"
    land_path = "../data/ne_10m_land/ne_10m_land.shp"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # find closes grid point for each station
    stations = tide_gauge_station.read_and_create_stations("../data/rlr_monthly/filelist.txt",
                                                           os.path.join(output_path, "metadata.txt"))
    print(f"Read {len(stations)} stations")
    # remove stations that are not present between 2010 and 2020
    stations_to_remove = []
    for station in list(stations.values()):
        present = False
        timeseries = station.timeseries
        for key in list(timeseries.keys()):
            if 2010 <= key <= 2020:
                present = True
                break
        if not present:
            stations_to_remove.append(station.id)
    for station in stations_to_remove:
        stations.pop(station)
    print(f"Stations present after 2010")
    # select stations that are between 9 degrees latitude and 15 degrees latitude and 53 degrees longitude and 55
    # degrees longitude (ostsee)
    stations = tide_gauge_station.detrend_and_mean_center_timeseries(stations)
    # selected_stations = []
    # for station in stations.values():
    #     if (9 <= station.longitude <= 15) and (50 <= station.latitude <= 56):
    #         selected_stations.append(station)
    nordsee = [stations[455], stations[468], stations[489], stations[413], stations[470], stations[20],
               stations[1551], stations[22], stations[9], stations[471], stations[32], stations[23]]

    golf_bengalen = [stations[2194], stations[1072], stations[1308], stations[205], stations[414],
                     stations[1161], stations[1270],
                     stations[543], stations[369], stations[2196], stations[2275]]

    radii = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for radius in radii:
        output_path = f"../output/analyze_clustering/golf_bengalen/"
        selected_stations = golf_bengalen
        print(f"Selected {len(selected_stations)} stations")
        # plot selected stations
        plot_timelines_and_save(selected_stations, output_path, land_path)
        # read in clusters

        with open(f"../output/RMS/PSMSL/10years/2010_2020/solution_sealevel{radius}.json", "r") as file:
            clusters = json.load(file)

        plot_clusters(selected_stations, output_path, land_path, clusters, stations, radius)

        output_path = "../output/analyze_clustering/nordsee/"
        selected_stations = nordsee
        print(f"Selected {len(selected_stations)} stations")
        # plot selected stations
        plot_timelines_and_save(selected_stations, output_path, land_path)
        plot_clusters(selected_stations, output_path, land_path, clusters, stations, radius)
        plot_graphs(selected_stations, output_path, land_path, clusters, stations, radius)
        plot_squares(selected_stations, output_path, land_path, clusters, stations, radius)


def plot_clusters(current_stations, output_dir, land_dir, current_clusters, all_stations, current_radius):
    """
    Plots the clusters of the selected stations
    :param all_stations:
    :param current_stations:
    :param output_dir:
    :param land_dir:
    :param current_clusters:
    :return:
    """
    fig, ax = plt.subplots(figsize=(40, 24))
    # colors = sea_level_line_graph.random_color_generator(len(current_clusters) + 100)
    colors = ["hotpink", "teal", "forestgreen", "darkblue", "purple", "blue",
              "hotpink", "violet", "lightgreen", "gold", "aqua", "coral", "crimson",
              "cyan", "darkgoldenrod", "darkorchid",
              "darkseagreen", "forestgreen", "gold", "indigo", "khaki", "lightcoral", "lightcyan",
              "lightgoldenrodyellow", "lavender", "lawngreen", "lemonchiffon", "magenta", "olive", "palevioletred"]
    counter = 0
    # check if stations are in a cluster together
    for center in current_clusters:
        for station_id in current_clusters[center]:
            if station_id in [station.id for station in current_stations]:
                print(f"Station {station_id} is in cluster {center}")

    for center in current_clusters:
        empty_center = True
        for station_id in current_clusters[center]:
            if not station_id in [station.id for station in current_stations]:
                continue
            empty_center = False
            station = all_stations[station_id]
            dates_to_remove = []
            for date in station.timeseries:
                if station.timeseries[date] == -99999 or date < 2010 or date > 2020:
                    dates_to_remove.append(date)
            for date in dates_to_remove:
                station.timeseries.pop(date)
                station.timeseries_detrended_normalized.pop(date)
            ax.plot(*zip(*sorted(station.timeseries.items())), label=station.id, color=colors[counter],
                    linewidth=4)
        if not empty_center:
            counter += 1
    print(counter)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"selected_stations_timeline_2010_to_2020_clusters{current_radius}.svg"))
    plt.close()
    pass


def plot_pcs(eof_dataset, eof_dates, eof_pcs, out_dir, sorted_dates, sorted_pcs):
    """
    Plot the PCs for the reconstructed data and the altimetry data
    :param eof_dataset:
    :param eof_dates:
    :param eof_pcs:
    :param out_dir:
    :param sorted_dates:
    :param sorted_pcs:
    :return:
    """
    # plot the PCs for the reconstructed data and the altimetry data
    for i in range(eof_dataset.eof.size):
        fig, ax = plt.subplots()
        ax.plot(eof_dates, eof_pcs[:, i], label="EOFs")
        ax.plot(sorted_dates, [pc[i][0] for pc in sorted_pcs], label="Reconstructed")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel("Time")
        ax.set_ylabel("PCs")
        # only show every 5th year on x-axis
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1825))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.title(f"PCs for EOF {i}")
        ax.legend()
        plt.savefig(os.path.join(out_dir, f"PCs_{i}.svg"), dpi=600)
        plt.close(fig)


def plot_line_graph_and_regions(current_output_dir: str, line_graph_gdf: geopandas.GeoDataFrame, regions,
                                land_path: str,
                                stations_gdf: geopandas.GeoDataFrame):
    """
    Plot the line graph and the regions on the map
    :param line_graph_gdf:
    :param stations_gdf:
    :param current_output_dir:
    :param regions:
    :param land_path:
    :return:
    """
    land_gdf = geopandas.read_file(land_path)

    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    regions.plot(ax=ax, color="aliceblue", zorder=4, linewidth=4, alpha=0.05)
    regions.boundary.plot(ax=ax, color="blue", zorder=5, linewidth=0.5, alpha=0.2)
    if "color" in stations_gdf.columns:
        stations_gdf.plot(ax=ax, color=stations_gdf["color"], markersize=5, zorder=3)
    else:
        stations_gdf.plot(ax=ax, color="green", markersize=5, zorder=3)
    line_graph_gdf.plot(ax=ax, color=line_graph_gdf["color"], zorder=2, linewidth=3)
    plt.xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks([-90, -45, 0, 45, 90])
    plt.savefig(os.path.join(current_output_dir, f"voronoi_sections_graph.svg"))
    plt.close()
    return
