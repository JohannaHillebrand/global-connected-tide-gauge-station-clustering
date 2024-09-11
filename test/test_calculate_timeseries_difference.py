from unittest import TestCase

from src.inner.tide_gauge_station import TideGaugeStation
from src.inner.timeseries_difference import calculate_rms_difference_between_pairs_of_stations, \
    subtract_mean_from_timeseries, calculate_difference_between_all_pairs_of_stations, \
    remove_dates_before_and_after_threshold


class Test(TestCase):
    def test_calculate_rms_difference_between_pairs_of_stations(self):
        timeline_a = {2020.1: 10, 2020.2: 20, 2020.3: 30}
        timeline_b = {2020.1: 5, 2020.2: 15, 2020.3: 25}
        station_a = TideGaugeStation(1, "A", 0, 0, timeline_a, timeline_a)
        station_b = TideGaugeStation(2, "B", 0, 0, timeline_b, timeline_b)
        overlap, gap_counter_result, difference_result, percentage_a_by_b, percentage_b_by_a = (
            calculate_rms_difference_between_pairs_of_stations(
                station_a,
                station_b, 'test_output/metadata.txt'))
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        self.assertEqual(overlap, 3)
        assert gap_counter_result == 0
        assert difference_result == 5

    def test_calculate_rms_difference_between_pairs_of_stations_with_gap(self):
        timeline_a = {2020.1: 10, 2020.2: 20, 2020.3: 30}
        timeline_b = {2020.1: 5, 2020.2: 15, 2020.3: 25, 2020.4: 35}
        station_a = TideGaugeStation(1, "A", 0, 0, timeline_a, timeline_a)
        station_b = TideGaugeStation(2, "B", 0, 0, timeline_b, timeline_b)
        overlap, gap_counter_result, difference_result, percentage_a_by_b, percentage_b_by_a = (
            calculate_rms_difference_between_pairs_of_stations(
                station_a,
                station_b, 'test_output/metadata.txt'))
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        assert gap_counter_result == 1
        assert difference_result == 5

    def test_calculate_rms_difference_between_pairs_of_stations_with_gap_2(self):
        timeline_a = {2020.1: 10, 2020.2: 20, 2020.3: 30, 2020.4: 40}
        timeline_b = {2020.1: 5, 2020.3: 25, 2020.4: 35}
        station_a = TideGaugeStation(1, "A", 0, 0, timeline_a, timeline_a)
        station_b = TideGaugeStation(2, "B", 0, 0, timeline_b, timeline_b)
        overlap, gap_counter_result, difference_result, percentage_a_by_b, percentage_b_by_a = (
            calculate_rms_difference_between_pairs_of_stations(
                station_a,
                station_b, 'test_output/metadata.txt'))
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        assert gap_counter_result == 1
        assert difference_result == 5

    def test_calculate_rms_difference_between_pairs_of_stations_with_missing_value(self):
        timeline_a = {2020.1: 10, 2020.2: 20, 2020.3: 30, 2020.4: -99999}
        timeline_b = {2020.1: 5, 2020.2: 15, 2020.3: 25, 2020.4: 35}
        station_a = TideGaugeStation(1, "A", 0, 0, timeline_a, timeline_a)
        station_b = TideGaugeStation(2, "B", 0, 0, timeline_b, timeline_b)
        overlap, gap_counter_result, difference_result, percentage_a_by_b, percentage_b_by_a = (
            calculate_rms_difference_between_pairs_of_stations(station_a, station_b, 'test_output/metadata.txt'))
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        assert gap_counter_result == 1
        assert difference_result == 5

    def test_calculate_rms_difference_between_pairs_of_stations_with_missing_value_2(self):
        timeline_a = {2020.1: 10, 2020.2: 20, 2020.3: 30, 2020.4: 40}
        timeline_b = {2020.1: 5, 2020.2: -99999, 2020.3: 25, 2020.4: -99999}
        station_a = TideGaugeStation(1, "A", 0, 0, timeline_a, timeline_a)
        station_b = TideGaugeStation(2, "B", 0, 0, timeline_b, timeline_b)
        overlap, gap_counter_result, difference_result, percentage_a_by_b, percentage_b_by_a = (
            calculate_rms_difference_between_pairs_of_stations(
                station_a,
                station_b, 'test_output/metadata.txt'))
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        assert gap_counter_result == 2
        assert difference_result == 5

    def test_calculate_rms_difference_between_pairs_of_stations_with_no_overlapping_values(self):
        timeline_a = {2020.1: 10, 2020.2: 20, 2020.3: 30, 2020.4: 40}
        timeline_b = {2020.5: 5, 2020.6: 15, 2020.7: 25, 2020.8: 35}
        station_a = TideGaugeStation(1, "A", 0, 0, timeline_a, timeline_a)
        station_b = TideGaugeStation(2, "B", 0, 0, timeline_b, timeline_b)
        overlap, gap_counter_result, difference_result, percentage_a_by_b, percentage_b_by_a = (
            calculate_rms_difference_between_pairs_of_stations(
                station_a,
                station_b, 'test_output/metadata.txt'))
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        assert gap_counter_result == 8
        assert difference_result == float('inf')

    def test_calculate_rms_difference_between_pairs_of_stations_with_real_values(self):
        station_680 = TideGaugeStation(680, "A", 0, 0, {}, {})
        with open('../data/rlr_monthly/data/680.rlrdata', 'r') as file:
            for line in file:
                split_line = line.split(";")
                date = float(split_line[0].strip())
                sea_level = float(split_line[1].strip())
                station_680.timeseries[date] = sea_level
        station_681 = TideGaugeStation(681, "B", 0, 0, {}, {})
        with open('../data/rlr_monthly/data/681.rlrdata', 'r') as file:
            for line in file:
                split_line = line.split(";")
                date = float(split_line[0].strip())
                sea_level = float(split_line[1].strip())
                station_681.timeseries[date] = sea_level
        stations = {680: station_680, 681: station_681}
        stations = subtract_mean_from_timeseries(stations)

        overlap, gap_counter_result, difference_result, percentage_a_by_b, percentage_b_by_a = (
            calculate_rms_difference_between_pairs_of_stations(
                stations[680],
                stations[681], 'test_output/metadata.txt'))
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        self.assertEqual(gap_counter_result, 47)
        self.assertEqual(difference_result, 23.542662928590676)

    def test_calculate_rms_difference_between_pairs_of_stations_with_real_values2(self):
        station2401 = TideGaugeStation(2401, "A", 0, 0, {}, {})
        with open('../data/rlr_monthly/data/2401.rlrdata', 'r') as file:
            for line in file:
                split_line = line.split(";")
                date = float(split_line[0].strip())
                sea_level = float(split_line[1].strip())
                station2401.timeseries[date] = sea_level
        station2402 = TideGaugeStation(2402, "B", 0, 0, {}, {})
        with open('../data/rlr_monthly/data/2402.rlrdata', 'r') as file:
            for line in file:
                split_line = line.split(";")
                date = float(split_line[0].strip())
                sea_level = float(split_line[1].strip())
                station2402.timeseries[date] = sea_level
        stations = {2401: station2401, 2402: station2402}
        stations = subtract_mean_from_timeseries(stations)
        difference_result = calculate_difference_between_all_pairs_of_stations(stations,
                                                                               'test_output/metadata.txt',
                                                                               True, False)
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        print("counter etc")
        print(difference_result)
        self.fail("Finish this test")

    def test_remove_dates_before_and_after_threshold(self):
        timeline1 = {2020.1: 10, 2021.2: 20, 2022.3: 30, 2023.4: 40}
        timeline2 = {2020.1: 5, 2020.2: 15, 2022.3: 25, 2023.4: 35}
        station1 = TideGaugeStation(1, "A", 0, 0, timeline1, {})
        station2 = TideGaugeStation(2, "B", 0, 0, timeline2, {})
        stations = {1: station1, 2: station2}
        start_year = 2021
        end_year = 2023
        result = remove_dates_before_and_after_threshold(start_year, end_year, stations, 'test_output/metadata.txt')
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        expected1 = station1
        expected1.timeseries = {2021.2: 20, 2022.3: 30}
        expected1.timeseries_detrended_normalized = {}
        expected2 = station2
        expected2.timeseries = {2022.3: 25}
        expected2.timeseries_detrended_normalized = {}
        expected = {1: expected1, 2: expected2}

        assert result == expected, (f'expected {expected}, got {result}')

    def test_subtract_mean_from_timeseries(self):
        station_a = TideGaugeStation(1, "A", 0, 0, {2020.1: 10, 2020.2: 20, 2020.3: 30}, {})
        station_b = TideGaugeStation(2, "B", 0, 0, {2020.1: 5, 2020.2: 15, 2020.3: 25}, {})
        stations = {1: station_a, 2: station_b}
        result = subtract_mean_from_timeseries(stations)
        expected_a = TideGaugeStation(1, "A", 0, 0, {2020.1: 10, 2020.2: 20, 2020.3: 30},
                                      {2020.1: -10, 2020.2: 0, 2020.3: 10})

        expected_b = TideGaugeStation(2, "B", 0, 0, {2020.1: 5, 2020.2: 15, 2020.3: 25},
                                      {2020.1: -10, 2020.2: 0, 2020.3: 10})
        expected = {1: expected_a, 2: expected_b}
        assert result == expected, (f'expected {expected}, got {result}')

    def test_subtract_mean_from_timeseries_with_missing_values(self):
        station_a = TideGaugeStation(1, "A", 0, 0, {2020.1: 10, 2020.2: 20, 2020.3: 30, 2020.4: -99999}, {})
        station_b = TideGaugeStation(2, "B", 0, 0, {2020.1: 5, 2020.2: 15, 2020.3: 25, 2020.4: 35}, {})
        station_c = TideGaugeStation(3, "C", 0, 0, {}, {})
        stations = {1: station_a, 2: station_b, 3: station_c}

        result = subtract_mean_from_timeseries(stations)

        expected_a = TideGaugeStation(1, "A", 0, 0, {2020.1: 10, 2020.2: 20, 2020.3: 30, 2020.4: -99999},
                                      {2020.1: -10, 2020.2: 0, 2020.3: 10, 2020.4: -99999})
        expected_b = TideGaugeStation(2, "B", 0, 0, {2020.1: 5, 2020.2: 15, 2020.3: 25, 2020.4: 35},
                                      {2020.1: -15, 2020.2: -5, 2020.3: 5, 2020.4: 15})
        expected = {1: expected_a, 2: expected_b}

        assert result == expected, (f'expected {expected}, got {result}')

    def test_calculate_rms_difference_between_pairs_of_stations_compare_identicals(self):
        station_a = TideGaugeStation(1, "A", 0, 0, {2020.1: 10, 2020.2: 20, 2020.3: 30}, {})
        station_b = TideGaugeStation(2, "B", 0, 0, {2020.1: 10, 2020.2: 20, 2020.3: 30}, {})
        stations = {1: station_a, 2: station_b}
        stations_normalized = subtract_mean_from_timeseries(stations)
        overlap, gap_counter_result, difference_result, percentage_a_by_b, percentage_b_by_a = (
            calculate_rms_difference_between_pairs_of_stations(
                stations_normalized[1],
                stations_normalized[2], 'test_output/metadata.txt'))
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        assert overlap == 3
        assert gap_counter_result == 0
        assert difference_result == 0
        assert percentage_a_by_b == 100
        assert percentage_b_by_a == 100

    def test_calculate_rms_difference_between_all_pairs_of_stations_compare_incomparable(self):
        station_a = TideGaugeStation(1, "A", 0, 0, {2020.1: 10, 2020.2: 20, 2020.3: 30}, {})
        station_b = TideGaugeStation(2, "B", 0, 0, {1920.1: 5, 1920.2: 15, 1920.3: 25, 1920.4: 35}, {})
        stations = {1: station_a, 2: station_b}
        stations_normalized = subtract_mean_from_timeseries(stations)
        difference_result = calculate_difference_between_all_pairs_of_stations(stations_normalized,
                                                                               'test_output/metadata.txt',
                                                                               True, False)
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        expected = {1: {2: (0, float('inf'))}, 2: {1: (0, float('inf'))}}
        assert difference_result == expected, f'expected {expected}, got {difference_result}'

    def test_remove_dates_before_and_after_threshold_with_stations_to_remove(self):
        station_a = TideGaugeStation(1, "A", 0, 0, {2020.1: 10, 2021.2: 15, 2022.3: 25, 2023.4: 40}, {})
        station_b = TideGaugeStation(2, "B", 0, 0, {2020.1: 5, 2021.2: 10, 2022.3: 30, 2023.4: 35}, {})
        station_c = TideGaugeStation(3, "C", 0, 0, {1920.1: 5, 1920.2: 15, 1922.3: 25, 1923.4: 35}, {})
        stations = {1: station_a, 2: station_b, 3: station_c}
        start_year = 2021
        end_year = 2023
        result = remove_dates_before_and_after_threshold(start_year, end_year, stations,
                                                         'test_output/metadata.txt')
        # empty metadata file
        open('test_output/metadata.txt', 'w').close()
        expected_a = TideGaugeStation(1, "A", 0, 0, {2021.2: 15, 2022.3: 25}, {})
        expected_b = TideGaugeStation(2, "B", 0, 0, {2021.2: 10, 2022.3: 30}, {})
        expected = {1: expected_a, 2: expected_b}

        assert result == expected, (f'expected {expected}, got {result}')
