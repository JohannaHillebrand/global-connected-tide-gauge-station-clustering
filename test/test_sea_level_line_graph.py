from unittest import TestCase

import networkx
from shapely import Point

from src.inner.sea_level_line_graph import group_stations, sort_neighbors_for_nodes, \
    find_closest_end_node, merge_line_graphs_that_are_close, sort_edges_to_add, create_line_graph
from src.inner.tide_gauge_station import TideGaugeStation


class Test(TestCase):
    def test_group_stations_compare180_with_180(self):
        # dist 1-2 = 8986.25 km
        # dist 1-3 = 6045.69 km
        # dist 2-3 = 4131.91 km
        # dist 1-4 = 5748.49 km
        # dist 2-4 = 4333.64 km
        # dist 3-4 = 297.64 km
        cologne = TideGaugeStation(1, 'name', 50.95, 6.95, {}, {})
        san_francisco = TideGaugeStation(2, 'name', 37.77, -122.41, {}, {})
        new_york = TideGaugeStation(3, 'name', 40.71, -74.01, {}, {})
        boston = TideGaugeStation(4, 'name', 42.36, -71.06, {}, {})
        stations = [cologne, san_francisco, new_york, boston]
        expected = {1: [], 2: [], 3: [boston], 4: [new_york]}
        result = group_stations(stations, 'test_output/metadata.txt')
        open('test_output/metadata.txt', 'w').close()
        assert result == expected, (f'Expected {expected}, but got {result}')

    def test_sort_neighbors_for_nodes(self):
        new_york = TideGaugeStation(3, 'name', 40.71, -74.01, {}, {})
        boston = TideGaugeStation(4, 'name', 42.36, -71.06, {}, {})
        providence = TideGaugeStation(5, 'name', 41.82, -71.41, {}, {})
        stations_with_groups = {1: [], 2: [], 3: [boston, providence], 4: [new_york, providence], 5: [new_york, boston]}
        sealevel_difference = {1: {2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4},
                               2: {1: 0.1, 3: 0.2, 4: 0.3, 5: 0.4},
                               3: {1: 0.1, 2: 0.2, 4: 0.1956, 5: 0.0673423},
                               4: {1: 0.1, 2: 0.2, 3: 0.1956, 5: 0.123},
                               5: {1: 0.1, 2: 0.2, 3: 0.0673423, 4: 0.123}}
        expected = {1: [], 2: [], 3: [[0.0673423, 5], [0.1956, 4]],
                    4: [[0.123, 5], [0.1956, 3]], 5: [[0.0673423, 3], [0.123, 4]]}

        result = sort_neighbors_for_nodes(sealevel_difference, stations_with_groups)
        assert result == expected, (f'Expected {expected}, but got {result}')

    def test_group_stations(self):
        station_1 = TideGaugeStation(1, 'name', 1, 7.5, {}, {})
        station_2 = TideGaugeStation(2, 'name', 1, 15, {}, {})
        station_3 = TideGaugeStation(3, 'name', 3, 7, {}, {})
        station_4 = TideGaugeStation(4, 'name', -3, -5, {}, {})
        stations = [station_1, station_2, station_3, station_4]
        expected = {1: [station_2, station_3], 2: [station_1, station_3],
                    3: [station_1, station_2], 4: []}
        result = group_stations(stations, "test_output/metadata.txt")
        open('test_output/metadata.txt', 'w').close()
        assert result == expected, (f'Expected {expected}, but got {result}')

    def test_find_closest_end_node(self):
        graph = networkx.Graph()
        graph.add_node(1, geometry=Point(1, 1))
        graph.add_node(2, geometry=Point(2, 2))
        graph.add_node(3, geometry=Point(3, 3))
        graph.add_node(4, geometry=Point(4, 4))
        graph.add_node(5, geometry=Point(5, 5))
        graph.add_edge(1, 2)
        graph.add_edge(4, 3, )
        graph.add_edge(4, 5)
        edge_nodes = [1, 2, 3, 5]
        result = find_closest_end_node([1, graph, edge_nodes])
        expected = [1, 3]
        assert result == expected, (f'Expected {expected}, but got {result}')

    def test_merge_line_graphs_that_are_close(self):
        graph = networkx.Graph()
        graph.add_node(1, geometry=Point(1, 1))
        graph.add_node(2, geometry=Point(2, 2))
        graph.add_node(3, geometry=Point(3, 3))
        graph.add_node(4, geometry=Point(4, 4))
        graph.add_node(5, geometry=Point(5, 5))
        graph.add_node(6, geometry=Point(6, 6))
        graph.add_node(7, geometry=Point(7, 7))
        graph.add_edge(1, 2)
        graph.add_edge(4, 3)
        graph.add_edge(4, 5)
        graph.add_edge(7, 6)
        sea_level_differences = {1: {2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.4, 7: 0.5},
                                 2: {1: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.4, 7: 0.5},
                                 3: {1: 0.2, 2: 0.2, 4: 0.1, 5: 0.4, 6: 0.4, 7: 0.5},
                                 4: {1: 0.1, 2: 0.2, 3: 0.1956, 5: 0.4, 6: 0.4, 7: 0.5},
                                 5: {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.4, 6: 0.1, 7: 0.2},
                                 6: {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4, 5: 0.1, 7: 0.1},
                                 7: {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.2, 6: 0.1}}
        result = merge_line_graphs_that_are_close(graph, sea_level_differences)
        expected = networkx.Graph()
        expected.add_node(1, geometry=Point(1, 1))
        expected.add_node(2, geometry=Point(2, 2))
        expected.add_node(3, geometry=Point(3, 3))
        expected.add_node(4, geometry=Point(4, 4))
        expected.add_node(5, geometry=Point(5, 5))
        expected.add_node(6, geometry=Point(6, 6))
        expected.add_node(7, geometry=Point(7, 7))
        expected.add_edge(1, 2)
        expected.add_edge(1, 3)
        expected.add_edge(3, 4)
        expected.add_edge(4, 5)
        expected.add_edge(5, 6)
        expected.add_edge(6, 7)
        assert result.nodes == expected.nodes, (f'Expected {expected.nodes}, but got {result.nodes}')
        assert result.edges == expected.edges, (f'Expected {expected.edges}, but got {result.edges}')
        assert (networkx.degree(result, node) <= 2 for node in result.nodes)

    def test_sort_edges_to_add(self):
        graph = networkx.Graph()
        graph.add_node(1, geometry=Point(1, 1))
        graph.add_node(2, geometry=Point(2, 2))
        graph.add_node(3, geometry=Point(3, 3))
        graph.add_node(4, geometry=Point(4, 4))
        graph.add_node(5, geometry=Point(5, 5))
        graph.add_node(6, geometry=Point(6, 6))
        graph.add_node(7, geometry=Point(7, 7))
        graph.add_edge(1, 2)
        graph.add_edge(4, 3)
        graph.add_edge(4, 5)
        graph.add_edge(7, 6)
        edges_to_add = [(5, 6), (1, 3), (2, 3), (5, 7), (1, 4), (2, 4), (1, 5), (1, 6), (2, 5), (2, 6), (3, 6),
                        (4, 6), (1, 7), (2, 7), (3, 7), (4, 7), (3, 4)]
        sea_level_differences = {1: {2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.4, 7: 0.5},
                                 2: {1: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.4, 7: 0.5},
                                 3: {1: 0.2, 2: 0.2, 4: 0.1, 5: 0.4, 6: 0.4, 7: 0.5},
                                 4: {1: 0.3, 2: 0.3, 3: 0.1, 5: 0.4, 6: 0.4, 7: 0.5},
                                 5: {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4, 6: 0.2, 7: 0.3},
                                 6: {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4, 5: 0.2, 7: 0.1},
                                 7: {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.3, 6: 0.1}}
        result = sort_edges_to_add(edges_to_add, graph, sea_level_differences)
        expected = [(3, 4), (5, 6), (1, 3), (2, 3), (5, 7), (1, 4), (2, 4), (1, 5), (1, 6), (2, 5), (2, 6), (3, 6),
                    (4, 6), (1, 7), (2, 7), (3, 7), (4, 7)]

        assert result == expected, (f'Expected {expected}, but got {result}')

    def test_create_line_graph(self):
        station1 = TideGaugeStation(1, 'name', 1, 2, {}, {})
        station2 = TideGaugeStation(2, 'name', 1, 3, {}, {})
        station3 = TideGaugeStation(3, 'name', 3, 7, {}, {})
        station4 = TideGaugeStation(4, 'name', 4, 4, {}, {})
        station5 = TideGaugeStation(5, 'name', 5, 5, {}, {})
        stations = [station1, station2, station3, station4, station5]
        stations_with_groups = {1: [station2, station3, station4, station5],
                                2: [station1, station3, station4, station5],
                                3: [station1, station2, station4, station5],
                                4: [station1, station2, station3, station5],
                                5: [station1, station2, station3, station4]}
        sealevel_difference = {1: {2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3}, 2: {1: 0.1, 3: 0.2, 4: 0.3, 5: 0.3},
                               3: {1: 0.1, 2: 0.2, 4: 0.1956, 5: 0.4}, 4: {1: 0.1, 2: 0.2, 3: 0.1, 5: 0.1},
                               5: {1: 0.1, 2: 0.2, 3: 0.1, 4: 0.1}}
        expected = networkx.Graph()
        expected.add_node(1, geometry=Point(1, 1))
        expected.add_node(2, geometry=Point(2, 2))
        expected.add_node(3, geometry=Point(3, 3))
        expected.add_node(4, geometry=Point(4, 4))
        expected.add_node(5, geometry=Point(5, 5))
        expected.add_edge(1, 2)
        expected.add_edge(2, 3)
        expected.add_edge(1, 4)
        expected.add_edge(4, 5)
        result = create_line_graph(stations_with_groups, sealevel_difference, stations, "test_output/metadata.txt")
        open('test_output/metadata.txt', 'w').close()
        # assert result.nodes == expected.nodes, (f'Expected {expected.nodes}, but got {result.nodes}')
        assert result.edges == expected.edges, (f'Expected {expected.edges}, but got {result.edges}')
