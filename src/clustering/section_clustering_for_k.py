import os

import shapely

from src.inner import tide_gauge_station
from src.inner.cluster_sections import fill_region_dict, divide_and_cluster


def start(regions: {str: shapely.Polygon}, station_path: str, time_steps: [(int, int)], land_path: str, output_dir: str,
          wanted_number_of_centers: int):
    """
    Start the clustering of the sections
    Given a certain number of stations, calculate the appropriate number of centers for each region (based on the
    regions area), calculate the solution for every region and create the final solution for this.
    :param regions:
    :param wanted_number_of_centers:
    :param output_dir:
    :param station_path:
    :param time_steps:
    :param land_path:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metadata_path = os.path.join(output_dir, "metadata.txt")
    stations = tide_gauge_station.read_and_create_stations(station_path, metadata_path)
    # Each section is defined by degree values
    regions_dict = fill_region_dict(regions)

    for time_step in time_steps:
        start_year = time_step[0]
        end_year = time_step[1]
        time_step_str = f"{start_year}_{end_year}"

        current_output_dir = os.path.join(output_dir, time_step_str)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        metadata_path = os.path.join(current_output_dir, "metadata.txt")
        with open(metadata_path, "w") as file:
            file.write(f"------------------------------------------ \n\n")
            file.write(f"Start for time step: {time_step}\n")
        # filter stations to use only the ones present in the current timestep
        stations_for_time_step = tide_gauge_station.filter_stations_for_time_step(stations, start_year, end_year)
        # print(f"Number of stations for time step {time_step}: {len(stations_for_time_step)}")
        divide_and_cluster(current_output_dir, land_path, metadata_path, regions_dict,
                           stations_for_time_step, wanted_number_of_centers)

    return
