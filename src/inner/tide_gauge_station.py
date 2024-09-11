import glob
from dataclasses import dataclass

import geopandas
import scipy
import shapely


@dataclass
class TideGaugeStation:
    id: int
    name: str
    latitude: float
    longitude: float
    # key: date, value: water level
    timeseries: dict
    # subtract the mean from each value in the timeseries
    timeseries_detrended_normalized: dict
    area: float

    def __eq__(self, other):
        return (self.id == other.id and self.name == other.name and self.latitude == other.latitude and self.longitude
                == other.longitude and self.timeseries == other.timeseries and self.timeseries_detrended_normalized ==
                other.timeseries_detrended_normalized)


def read_and_create_stations(path: str, metadata_path: str):
    """
    Read the filelist.txt file and create a station object with the corresponding timeseries for each station in the
    file, the field detrended_normalized_timeseries is left empty and will be filled later
    :param metadata_path:
    :return:
    :param path: str
    """
    current_stations = {}
    counter = 0
    flag_counter = 0
    with open(path, "r") as file:
        for line in file:
            counter += 1
            split_line = line.split("; ")
            station_id = int(split_line[0])
            station_latitude = float(split_line[1].strip())
            station_longitude = float(split_line[2].strip())
            station_name = split_line[3].strip()
            station_timeseries = {}
            station_timeseries_normalized = {}
            current_station = TideGaugeStation(station_id, station_name, station_latitude, station_longitude,
                                               station_timeseries, station_timeseries_normalized, None)
            current_stations[station_id] = current_station

        # read in timeseries data
        # ASSUMPTION: the timeseries data is in the same folder as the filelist.txt file
        # REPLACE the flagged values with -99999 (only flag for 011 - might be different to more than 1cm,
        # should not be used in long-term trend analysis)
        stations_to_remove = []
        valid_values = 0
        no_data_values = 0
        station_directory = path.split("filelist.txt")[0] + "data/"
        for station in current_stations.keys():
            try:
                station_path = glob.glob(f"{station_directory}{station}.rlrdata")[0]
                with open(station_path, "r") as file:
                    for line in file:
                        split_line = line.split(";")
                        date = float(split_line[0].strip())
                        sea_level = float(split_line[1].strip())
                        flag = split_line[3].strip()
                        if flag == "011" or flag == "001 " or flag == "010":
                            sea_level = -99999
                            flag_counter += 1
                        if sea_level == -99999:
                            no_data_values += 1
                        else:
                            valid_values += 1

                        current_stations[station].timeseries[date] = sea_level
            except:
                # logger.warning(f"No file found for station {station}")
                stations_to_remove.append(station)
        for station in stations_to_remove:
            current_stations.pop(station)
        with open(metadata_path, "w") as file:
            file.write(f"Found {counter} stations in the dataset\n")
            file.write(f"Found {flag_counter} flagged values in the dataset\n")
            file.write(f"Found {valid_values} valid data points in the dataset\n")
            file.write(f"Total number of data points in the dataset: {valid_values + flag_counter}\n")
            file.write(f"Percentage of flagged values: {flag_counter / (flag_counter + valid_values) * 100}\n")
            file.write(f"Found {no_data_values} missing data points in the dataset\n")
            file.write(f"Created {len(current_stations)} stations.\n")
    return current_stations


def filter_stations_for_time_step(stations: {int: TideGaugeStation}, start_year: float, end_year: float):
    """
    Filter the stations for the given time step
    :param stations:
    :param start_year:
    :param end_year:
    :return:
    """
    filtered_stations = {}
    for station in stations.keys():
        station_timeseries = stations[station].timeseries
        filtered_timeseries = {key: value for key, value in station_timeseries.items() if start_year <= key <= end_year}
        if filtered_timeseries:
            filtered_stations[station] = TideGaugeStation(stations[station].id, stations[station].name,
                                                          stations[station].latitude,
                                                          stations[station].longitude, filtered_timeseries, {},
                                                          stations[station].area)
    return filtered_stations


def gdf_from_stations(stations: {int: TideGaugeStation}):
    """
    Create a GeoDataFrame from the stations
    :param stations:
    :return:
    """
    station_dict = {"id": [], "geometry": []}
    for station in stations.values():
        station_dict["id"].append(station.id)
        station_dict["geometry"].append(shapely.Point(station.longitude, station.latitude))
    stations_gdf = geopandas.GeoDataFrame(station_dict, crs="EPSG:4326", geometry=station_dict["geometry"],
                                          index=station_dict["id"])
    return stations_gdf


def detrend_and_mean_center_timeseries(current_stations):
    """
    Remove the linear trend from each timeseries
    :param current_stations:
    :return:
    """
    for station in current_stations.values():
        dates = []
        sea_level = []
        sorted_timeseries = dict(sorted(station.timeseries.items()))
        for date in sorted_timeseries.keys():
            if station.timeseries[date] != -99999:
                dates.append(date)
                sea_level.append(station.timeseries[date])
        if len(dates) > 1:
            detrended_sea_level = scipy.signal.detrend(sea_level)
            sum_detrended_sea_level = sum(detrended_sea_level)
            avg_detrended_sea_level = sum_detrended_sea_level / len(detrended_sea_level)
            normalized_and_detrended_sea_level = [x - avg_detrended_sea_level for x in detrended_sea_level]
            for i in range(len(dates)):
                station.timeseries_detrended_normalized[dates[i]] = normalized_and_detrended_sea_level[i]
        for date in station.timeseries.keys():
            if station.timeseries[date] == -99999:
                station.timeseries_detrended_normalized[date] = -99999

    return current_stations


def mean_center_timeseries(stations: {int: TideGaugeStation}):
    """
    Mean center the timeseries
    :param stations:
    :return:
    """
    for station in stations.values():
        sum_all_values = 0
        number_of_valid_dates = 0
        for date in station.timeseries.keys():
            if not station.timeseries[date] == -99999:
                sum_all_values += station.timeseries[date]
                number_of_valid_dates += 1
        if number_of_valid_dates > 0:
            mean = sum_all_values / number_of_valid_dates
            for date in station.timeseries.keys():
                if not station.timeseries[date] == -99999:
                    station.timeseries_detrended_normalized[date] = station.timeseries[date] - mean
                else:
                    station.timeseries_detrended_normalized[date] = -99999
    return stations


def filter_timeseries_without_removing_stations(stations, start_year: int, end_year: int):
    """
    Filter the timeseries for the given time range
    :param end_year:
    :param start_year:
    :param stations:
    :return:
    """
    filtered_stations = {}
    for station in stations.values():
        filtered_timeseries = {key: value for key, value in station.timeseries_detrended_normalized.items()
                               if start_year <= key <= end_year}

        filtered_stations[station.id] = TideGaugeStation(station.id, station.name, station.latitude,
                                                         station.longitude, station.timeseries, filtered_timeseries,
                                                         station.area)

    return filtered_stations


def mean_center_for_every_ten_years(stations: {int: TideGaugeStation}):
    """
    Mean center the timeseries for every ten years
    :param stations:
    :return:
    """
    earliest_date = float("inf")
    latest_date = 0
    for station in stations.values():
        for date in station.timeseries.keys():
            if date < earliest_date:
                earliest_date = date
            if date > latest_date:
                latest_date = date
    for i in range(int(earliest_date), int(latest_date), 10):
        for station in stations.values():
            sum_all_values_in_interval = 0
            number_of_valid_dates_in_interval = 0
            for date in station.timeseries.keys():
                if i <= date < i + 10:
                    if not station.timeseries[date] == -99999:
                        sum_all_values_in_interval += station.timeseries[date]
                        number_of_valid_dates_in_interval += 1
            if number_of_valid_dates_in_interval > 0:
                mean = sum_all_values_in_interval / number_of_valid_dates_in_interval
                for date in station.timeseries.keys():
                    if i <= date < i + 10:
                        if not station.timeseries[date] == -99999:
                            station.timeseries_detrended_normalized[date] = station.timeseries[date] - mean
                        else:
                            station.timeseries_detrended_normalized[date] = -99999
    return stations
