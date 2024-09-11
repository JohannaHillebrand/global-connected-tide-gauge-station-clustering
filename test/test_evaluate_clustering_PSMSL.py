from unittest import TestCase

from src.inner import timeseries_difference


class Test(TestCase):
    def test_calculate_rms_all_stations(self):
        timeseries1 = {1992.0: 0.0, 1993.0: 1.0, 1994.0: 2.0}
        timeseries2 = {1992.0: 2.0, 1993.0: 6.0, 1994.0: 1.0}
        station1 = timeseries_difference.TideGaugeStation(1, "station1", 1.0, 1.0, timeseries1, None)
        station2 = timeseries_difference.TideGaugeStation(2, "station2", 1.0, 1.0, timeseries2, None)
        stations = {1: station1, 2: station2}
        global_sea_level = {1992.0: 8.0, 1993.0: 1.0, 1994.0: 6.0}
        # mean center the values
        current_sum = 0
        for value in global_sea_level.values():
            current_sum += value
        mean = current_sum / len(global_sea_level)
        for current_date in global_sea_level.keys():
            global_sea_level[current_date] = global_sea_level[current_date] - mean
        radii = {1: 1.0, 2: 1.0}
        rms = evaluate_clustering_PSMSL.calculate_rms_all_stations(stations, radii, global_sea_level)

        assert rms == {1: 4.02077936060494, 2: 4.02077936060494}, f"RMS: {rms}"
