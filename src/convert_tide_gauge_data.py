# read in tide gauges, convert measurements to meters, save aggregated in one csv
import csv
import os

from src.inner import tide_gauge_station
from src.inner.tide_gauge_station import TideGaugeStation


def write_to_file(path: str, tide_gauge_stations: dict[int, TideGaugeStation]):
    """
    write station data to file and convert to meters
    """
    # make csv file
    file_path = os.path.join(path, "tide_gauges.csv")
    # header station_id, station_name, first_date, second_date,..., final_date
    all_dates = set()
    for station in tide_gauge_stations.values():
        for date in station.timeseries:
            all_dates.add(date)
    sorted_dates = sorted(all_dates)
    with open(file_path, "w") as f:
        f.write(f"station_id, station_name, latitude, longitude")
        for date in sorted_dates:
            f.write(f", {date}")
        f.write("\n")
        for station in tide_gauge_stations.values():
            # remove "," from names
            if "," in station.name:
                station.name = station.name.replace(",", "-")
            f.write(f"{station.id}, {station.name}, {station.latitude}, {station.longitude}")
            for date in sorted_dates:
                if date in station.timeseries:
                    value = station.timeseries[date]
                    if value == -99999:
                        f.write(f", None")
                    else:
                        # convert value from mm to m
                        value = value * 0.001
                        f.write(f", {value}")
                else:
                    f.write(f", None")
            f.write("\n")
    return


def check_file(original_stations: dict[int, TideGaugeStation], path):
    """
    check conversion
    """
    stations_for_checking = {}
    with open(os.path.join(path, "tide_gauges.csv"), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        first_row = next(csv_reader)
        all_dates = []
        for element in first_row[4:]:
            all_dates.append(float(element))

        for row in csv_reader:
            station_id = int(row[0])
            station_name = row[1]
            station_latitude = float(row[2])
            station_longitude = float(row[3])
            timeseries = {}
            for date_index, element in enumerate(row[4:]):
                if element == " None":
                    timeseries[date_index] = -99999
                else:
                    timeseries[all_dates[date_index]] = float(element)

            current_station = TideGaugeStation(station_id, station_name, station_latitude, station_longitude,
                                               timeseries, {}, None)
            stations_for_checking[station_id] = current_station
    print(f"{len(stations_for_checking)} station after reading again")
    print(f"{len(original_stations)} before")

    # check if there are the same number of valid dates as before
    counter_check = 0
    for station_id, station in stations_for_checking.items():
        for element in station.timeseries.values():
            if element != -99999:
                counter_check += 1
    print(counter_check)
    counter_check = 0
    for station_id, station in original_stations.items():
        for element in station.timeseries.values():
            if element != -99999:
                counter_check += 1
    print(counter_check)


if __name__ == "__main__":
    current_path = "../data/rlr_monthly/filelist.txt"
    output_path = "../output/stations/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    station_info_path = os.path.join(output_path, "station_info.txt")
    stations = tide_gauge_station.read_and_create_stations(current_path, station_info_path)
    write_to_file(output_path, stations)
    # read csv to check if correct
    check_file(stations, output_path)
