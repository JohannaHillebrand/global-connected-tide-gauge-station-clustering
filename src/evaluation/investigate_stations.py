import os.path

import matplotlib.pyplot as plt

from src.inner import tide_gauge_station

if __name__ == "__main__":
    station_path = "../../data/rlr_monthly/filelist.txt"
    output_path = "../../output/investigate_stations/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # read in stations
    stations = tide_gauge_station.read_and_create_stations(station_path, os.path.join(output_path, "metadata.txt"))
    start_year = 1807
    end_year = 2024
    step_length = 10
    time_steps = [(i, i + step_length) for i in range(start_year, end_year, step_length)]
    existence = {}
    for station in stations.values():
        current_timeseries = station.timeseries
        for time_step in time_steps:
            if time_step not in existence:
                existence[time_step] = 0
            for year in current_timeseries.keys():
                if time_step[0] <= year <= time_step[1]:
                    existence[time_step] += 1
                    break

    with open(os.path.join(output_path, "existence.txt"), "w") as output_file:
        output_file.write(f"Overall number of stations: {len(stations)}\n")
        output_file.write("Number of stations per time step\n")
        for time_step, number in existence.items():
            output_file.write(f"{time_step}: {number}\n")
    # plot existing stations per time step
    plt.plot([time_step[0] for time_step in time_steps],
             [existence[time_step] for time_step in time_steps])
    plt.xticks([1807, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000, 2024])
    plt.xlabel("Time")
    plt.ylabel("Number of stations")
    plt.title("Number of stations per time step")
    plt.savefig(os.path.join(output_path, "existence.pdf"), dpi=300)
