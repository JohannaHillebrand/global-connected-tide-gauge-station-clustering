from unittest import TestCase

import networkx

from src.inner import line_graph_clustering
from src.inner.line_graph_clustering import compute_cluster_for_each_node, divide_graph_into_connected_components, \
    select_centers, \
    compute_clustering_for_given_radius, binary_search, compute_cluster_for_each_node_demanding_overlap


class Test(TestCase):

    def test_compute_subgraph_for_each_node_edge_node(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        radius = 2
        distance_dict = {1: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 2: {1: 1, 2: 0, 3: 1, 4: 4, 5: 6},
                         3: {1: 2, 2: 1, 3: 0, 4: 4, 5: 8}, 4: {1: 3, 2: 4, 3: 4, 4: 0, 5: 1},
                         5: {1: 4, 2: 6, 3: 8, 4: 1, 5: 0}}
        current_node = 1
        solution_graph = networkx.Graph()
        solution_graph.add_edge(1, 2)
        solution_graph.add_edge(2, 3)
        solution_belonging_dict = {1: [1], 2: [1], 3: [1]}

        result_graph, node_belonging_dict = compute_cluster_for_each_node(graph, radius, distance_dict, current_node,
                                                                          {})

        assert networkx.utils.graphs_equal(result_graph, solution_graph), (f"result should be {solution_graph}, "
                                                                           f"but is {result_graph}")
        assert node_belonging_dict == solution_belonging_dict, (
            f"result should be {solution_belonging_dict}, but is {node_belonging_dict}")

    def test_compute_subgraph_for_each_node_middle_node(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        radius = 2
        distance_dict = {1: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 2: {1: 1, 2: 0, 3: 1, 4: 4, 5: 6},
                         3: {1: 2, 2: 1, 3: 0, 4: 2, 5: 8}, 4: {1: 3, 2: 4, 3: 2, 4: 0, 5: 1},
                         5: {1: 4, 2: 6, 3: 8, 4: 1, 5: 0}}
        current_node = 3
        solution_graph = networkx.Graph()
        solution_graph.add_edge(1, 2)
        solution_graph.add_edge(2, 3)
        solution_graph.add_edge(3, 4)
        solution_belonging_dict = {1: [3], 2: [3], 3: [3], 4: [3]}

        result_graph, node_belonging_dict = compute_cluster_for_each_node(graph, radius, distance_dict, current_node,
                                                                          {})

        assert networkx.utils.graphs_equal(result_graph, solution_graph), (f"result should be {solution_graph}, "
                                                                           f"but is {result_graph}")

        assert node_belonging_dict == solution_belonging_dict, (
            f"result should be {solution_belonging_dict}, but is {node_belonging_dict}")

    def test_compute_subgraph_for_each_node_small_radius(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        radius = 0
        distance_dict = {1: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 2: {1: 1, 2: 0, 3: 1, 4: 4, 5: 6},
                         3: {1: 2, 2: 1, 3: 0, 4: 2, 5: 8}, 4: {1: 3, 2: 4, 3: 2, 4: 0, 5: 1},
                         5: {1: 4, 2: 6, 3: 8, 4: 1, 5: 0}}
        current_node = 3
        solution_graph = networkx.Graph()
        solution_graph.add_node(3)
        solution_belonging_dict = {3: [3]}

        result, node_belonging_dict = compute_cluster_for_each_node(graph, radius, distance_dict, current_node, {})

        assert networkx.utils.graphs_equal(result, solution_graph), (f"result should be {solution_graph}, "
                                                                     f"but is {result}")
        assert node_belonging_dict == solution_belonging_dict, (
            f"result should be {solution_belonging_dict}, but is {node_belonging_dict}")

    def test_compute_subgraph_for_each_node_no_neighbors(self):
        graph = networkx.Graph()
        graph.add_node(1)
        radius = 10
        distance_dict = {1: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 2: {1: 1, 2: 0, 3: 1, 4: 4, 5: 6},
                         3: {1: 2, 2: 1, 3: 0, 4: 2, 5: 8}, 4: {1: 3, 2: 4, 3: 2, 4: 0, 5: 1},
                         5: {1: 4, 2: 6, 3: 8, 4: 1, 5: 0}}
        current_node = 1
        solution_graph = networkx.Graph()
        solution_graph.add_node(1)
        solution_belonging_dict = {1: [1]}

        result_graph, node_belonging_dict = compute_cluster_for_each_node(graph, radius, distance_dict, current_node,
                                                                          {})

        assert networkx.utils.graphs_equal(result_graph, solution_graph), (f"result should be {solution_graph}, "
                                                                           f"but is {result_graph}")
        assert node_belonging_dict == solution_belonging_dict, (
            f"result should be {solution_belonging_dict}, but is {node_belonging_dict}")

    def test_divide_graph_into_connected_components(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        graph.add_edge(5, 6)
        graph.add_edge(6, 7)

        solution_graph1 = networkx.Graph()
        solution_graph1.add_edge(1, 2)
        solution_graph1.add_edge(2, 3)
        solution_graph1.add_edge(3, 4)
        solution_graph2 = networkx.Graph()
        solution_graph2.add_edge(5, 6)
        solution_graph2.add_edge(6, 7)
        solution_graphs = [solution_graph1, solution_graph2]

        result_graphs = divide_graph_into_connected_components(graph)

        assert networkx.utils.graphs_equal(result_graphs[0], solution_graphs[0]) and networkx.utils.graphs_equal(
            result_graphs[1], solution_graphs[1]), (
            f"result should be {solution_graphs[0]} and {solution_graphs[1]}, "
            f"but is {result_graphs[0]} and {result_graphs[1]}")

    def test_select_centers(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)
        possible_centers_dict = {1: networkx.Graph(), 2: networkx.Graph(), 3: networkx.Graph(), 4: networkx.Graph(),
                                 5: networkx.Graph(), 6: networkx.Graph()}
        possible_centers_dict[1].add_edge(1, 2)
        possible_centers_dict[1].add_edge(2, 3)
        possible_centers_dict[2].add_edge(1, 2)
        possible_centers_dict[2].add_edge(2, 3)
        possible_centers_dict[3].add_edge(1, 2)
        possible_centers_dict[3].add_edge(2, 3)
        possible_centers_dict[3].add_edge(3, 4)
        possible_centers_dict[4].add_edge(3, 4)
        possible_centers_dict[4].add_edge(4, 5)
        possible_centers_dict[4].add_edge(5, 6)
        possible_centers_dict[5].add_edge(4, 5)
        possible_centers_dict[5].add_edge(5, 6)
        possible_centers_dict[6].add_edge(4, 5)
        possible_centers_dict[6].add_edge(5, 6)
        node_belonging_dict = {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3, 4], 4: [3, 4, 5, 6], 5: [4, 5, 6], 6: [4, 5, 6]}

        expected_result_for_5 = possible_centers_dict[5].copy()
        expected_result_for_5.remove_node(4)

        solution_centers = {3: possible_centers_dict[3].copy(), 5: expected_result_for_5}

        result = select_centers(graph, possible_centers_dict, node_belonging_dict)

        assert networkx.utils.graphs_equal(result[3], solution_centers[
            3]), f"result should be {solution_centers[3]}, but is {result[3]}"

        assert networkx.utils.graphs_equal(result[5], solution_centers[
            5]), f"result should be {solution_centers[5]}, but is {result[5]}"

    def test_select_centers2(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)
        possible_centers_dict = {1: networkx.Graph(), 2: networkx.Graph(), 3: networkx.Graph(), 4: networkx.Graph(),
                                 5: networkx.Graph(), 6: networkx.Graph()}
        possible_centers_dict[1].add_edge(1, 2)
        possible_centers_dict[1].add_edge(2, 3)
        possible_centers_dict[2].add_edge(1, 2)
        possible_centers_dict[2].add_edge(2, 3)
        possible_centers_dict[3].add_edge(2, 3)
        possible_centers_dict[3].add_edge(3, 4)
        possible_centers_dict[4].add_edge(4, 5)
        possible_centers_dict[4].add_edge(5, 6)
        possible_centers_dict[5].add_edge(4, 5)
        possible_centers_dict[5].add_edge(5, 6)
        possible_centers_dict[6].add_edge(5, 6)

        node_belonging_dict = {1: [1, 2], 2: [1, 2, 3], 3: [2, 3], 4: [3, 4, 5], 5: [4, 5], 6: [4, 5, 6]}

        expected_result_for_4 = possible_centers_dict[4].copy()

        solution_centers = {1: possible_centers_dict[1].copy(), 4: expected_result_for_4}

        result = select_centers(graph, possible_centers_dict, node_belonging_dict)

        assert networkx.utils.graphs_equal(result[1], solution_centers[
            1]), f"result should be {solution_centers[1]}, but is {result[1]}"

        assert networkx.utils.graphs_equal(result[4], solution_centers[
            4]), f"result should be {solution_centers[4]}, but is {result[4]}"

    def test_compute_clustering_for_given_radius(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        graph.add_edge(5, 6)
        graph.add_edge(6, 7)
        graph.add_node(8)
        radius = 2
        distance_dict = {1: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 1, 7: 3, 8: 1},
                         2: {1: 1, 2: 0, 3: 1, 4: 4, 5: 6, 6: 8, 7: 6, 8: 1},
                         3: {1: 2, 2: 1, 3: 0, 4: 4, 5: 8, 6: 4, 7: 6, 8: 1},
                         4: {1: 3, 2: 4, 3: 4, 4: 0, 5: 1, 6: 4, 7: 8, 8: 1},
                         5: {1: 4, 2: 6, 3: 8, 4: 1, 5: 0, 6: 2, 7: 4, 8: 1},
                         6: {1: 1, 2: 8, 3: 4, 4: 4, 5: 2, 6: 0, 7: 2, 8: 1},
                         7: {1: 3, 2: 6, 3: 6, 4: 8, 5: 4, 6: 2, 7: 0, 8: 1},
                         8: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0}}

        graph_for_1 = networkx.Graph()
        graph_for_1.add_edge(1, 2)
        graph_for_1.add_edge(2, 3)
        graph_for_4 = networkx.Graph()
        graph_for_4.add_node(4)
        graph_for_6 = networkx.Graph()
        graph_for_6.add_edge(5, 6)
        graph_for_6.add_edge(6, 7)
        graph_for_8 = networkx.Graph()
        graph_for_8.add_node(8)
        expected_result = {1: graph_for_1, 4: graph_for_4, 6: graph_for_6, 8: graph_for_8}

        number_of_centers, result = compute_clustering_for_given_radius(graph, radius, distance_dict, False, 0)

        assert networkx.utils.graphs_equal(result[1], expected_result[
            1]), f"result should be {expected_result[1]}, but is {result[1]}"
        assert networkx.utils.graphs_equal(result[4], expected_result[
            4]), f"result should be {expected_result[4]}, but is {result[4]}"
        assert networkx.utils.graphs_equal(result[6], expected_result[
            6]), f"result should be {expected_result[6]}, but is {result[6]}"
        assert networkx.utils.graphs_equal(result[8], expected_result[8]), f"result should be {expected_result[8]}, " \
                                                                           f"but is {result[8]}"

    def test_binary_search(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        graph.add_edge(5, 6)
        graph.add_edge(6, 7)
        graph.add_node(8)

        distance_dict = {1: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 1, 7: 3, 8: 1},
                         2: {1: 1, 2: 0, 3: 1, 4: 4, 5: 6, 6: 8, 7: 6, 8: 1},
                         3: {1: 2, 2: 1, 3: 0, 4: 2, 5: 8, 6: 4, 7: 6, 8: 1},
                         4: {1: 3, 2: 4, 3: 2, 4: 0, 5: 1, 6: 4, 7: 16, 8: 1},
                         5: {1: 4, 2: 6, 3: 8, 4: 1, 5: 0, 6: 2, 7: 4, 8: 1},
                         6: {1: 1, 2: 8, 3: 4, 4: 4, 5: 2, 6: 0, 7: 2, 8: 1},
                         7: {1: 3, 2: 6, 3: 6, 4: 8, 5: 4, 6: 2, 7: 0, 8: 1},
                         8: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0}}

        result_distance, result_centers, result_k, number_of_iterations = binary_search(graph, 3, distance_dict, False,
                                                                                        0)

        assert result_distance == 2, f"result_distance should be 2, but is {result_distance}"
        assert result_k == 3, f"result_k should be 3, but is {result_k}"

    def test_binary_search_k_is_exactly_middle(self):
        g = networkx.Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        distance_dict = {1: {1: 0, 2: 2, 3: 4}, 2: {1: 2, 2: 0, 3: 2}, 3: {1: 4, 2: 2, 3: 0}}

        result_distance, result_centers, result_k, number_iterations = binary_search(g, 1, distance_dict, False, 0)

        assert result_distance == 2, f"result_distance should be 2, but is {result_distance}"
        assert result_k == 1, f"result_k should be 1, but is {result_k}"

    # def test_read_data(self):
    #     # def read_data(distance_file: str, edgelist_file: str, id_to_index: str):
    #     distance_file = "../data/distance_sealevel_mean.csv"
    #     edgelist_file = "../data/trees_edgelist.csv"
    #     id_to_index = "../data/idtoindex.json"
    #     read_data(distance_file, edgelist_file, id_to_index)
    #     self.fail()

    def test_compute_cluster_for_each_node_demanding_overlap(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        radius = 2
        distance_dict = {1: {1: [100, 0], 2: [95, 1], 3: [10, 2], 4: [20, 3]},
                         2: {1: [10, 1], 2: [100, 0], 3: [30, 1], 4: [90, 4]},
                         3: {1: [40, 2], 2: [40, 1], 3: [100, 0], 4: [95, 4]},
                         4: {1: [95, 3], 2: [50, 4], 3: [40, 4], 4: [100, 0]}}
        current_node = 1
        solution_graph = networkx.Graph()
        solution_graph.add_edge(1, 2)
        solution_belonging_dict = {1: [1], 2: [1]}

        result_graph, node_belonging_dict = compute_cluster_for_each_node_demanding_overlap(graph, radius,
                                                                                            distance_dict,
                                                                                            current_node,
                                                                                            {}, 90)

        assert networkx.utils.graphs_equal(result_graph, solution_graph), (f"result should be {solution_graph}, "
                                                                           f"but is {result_graph}")
        assert node_belonging_dict == solution_belonging_dict, (
            f"result should be {solution_belonging_dict}, but is {node_belonging_dict}")

    def test_compute_cluster_complete_clustering_with_overlap(self):
        graph = networkx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        radius = 2
        distance_dict = {1: {1: [100, 0], 2: [95, 1], 3: [10, 2], 4: [20, 3]},
                         2: {1: [10, 1], 2: [100, 0], 3: [30, 1], 4: [90, 4]},
                         3: {1: [40, 2], 2: [40, 1], 3: [100, 0], 4: [95, 2]},
                         4: {1: [95, 3], 2: [50, 4], 3: [40, 4], 4: [100, 0]}}
        current_node = 1

        solution_number_of_centers = 2
        graph1 = networkx.Graph()
        graph1.add_edge(1, 2)
        graph2 = networkx.Graph()
        graph2.add_edge(3, 4)
        solution = {1: graph1, 3: graph2}

        current_number_of_centers, all_center_nodes = compute_clustering_for_given_radius(graph, 2, distance_dict, True,
                                                                                          90)

        assert solution_number_of_centers == current_number_of_centers, (
            f"result should be {solution_number_of_centers}, but is {current_number_of_centers}")
        assert networkx.utils.graphs_equal(all_center_nodes[1], solution[
            1]), f"result should be {solution[1]}, but is {all_center_nodes[1]}"
        assert networkx.utils.graphs_equal(all_center_nodes[3], solution[
            3]), f"result should be {solution[4]}, but is {all_center_nodes[4]}"

    def test_for_non_disjoint_clusters(self):
        input_graph = networkx.Graph()
        input_graph.add_edge('a', 'b')
        input_graph.add_edge('b', 'c')
        input_graph.add_edge('c', 'd')
        input_graph.add_edge('d', 'e')
        input_graph.add_edge('e', 'f')
        distances = {'a': {'a': 0, 'b': 2, 'c': 2, 'd': 1, 'e': 2, 'f': 2},
                     'b': {'a': 2, 'b': 0, 'c': 2, 'd': 1, 'e': 2, 'f': 2},
                     'c': {'a': 2, 'b': 2, 'c': 0, 'd': 1, 'e': 1, 'f': 1},
                     'd': {'a': 1, 'b': 1, 'c': 1, 'd': 0, 'e': 2, 'f': 2},
                     'e': {'a': 2, 'b': 2, 'c': 1, 'd': 2, 'e': 0, 'f': 2},
                     'f': {'a': 2, 'b': 2, 'c': 1, 'd': 2, 'e': 2, 'f': 0}}
        radius = 1

        result_centers, result_center_nodes = compute_clustering_for_given_radius(input_graph, radius, distances, False,
                                                                                  0)

        d_graph = networkx.Graph()
        d_graph.add_edge('d', 'c')
        d_graph.add_edge('c', 'b')
        d_graph.add_edge('b', 'a')
        e_graph = networkx.Graph()
        e_graph.add_node('e')
        f_graph = networkx.Graph()
        f_graph.add_node('f')
        expected_graphs = {'d': d_graph, 'e': e_graph, 'f': f_graph}

        assert result_centers == 3, f"result_centers should be 3, but is {result_centers}"
        assert networkx.utils.graphs_equal(result_center_nodes['e'], expected_graphs['e']), (
            f"graph for e should be {expected_graphs['e']}, but is {result_center_nodes['e']}")
        assert networkx.utils.graphs_equal(result_center_nodes['f'], expected_graphs['f']), (f'graph for f should be '
                                                                                             f'{expected_graphs["f"]}, '
                                                                                             f'but is '
                                                                                             f'{result_center_nodes["f"]}')
        assert networkx.utils.graphs_equal(result_center_nodes['d'], expected_graphs['d']), (f'graph for d should be '
                                                                                             f'{expected_graphs["d"]}, '
                                                                                             f'but is {
                                                                                             result_center_nodes["d"]}')

    def test_line_graph_binary_search_for_given_graph(self):
        differences = {'284': {'820': 70.44516379665707, '826': 53.95942085476776, '986': 101.98113250797599},
                       '820': {'284': 70.44516379665707, '826': 52.996091297799246, '986': 120.37080223240574},
                       '826': {'284': 53.95942085476776, '820': 52.996091297799246, '986': 84.32094955062315},
                       '986': {'284': 101.98113250797599, '820': 120.37080223240574, '826': 84.32094955062315}}
        line_graph = networkx.Graph()
        line_graph.add_edge('826', '820')
        line_graph.add_edge('284', '986')
        line_graph.add_edge('284', '820')
        k = 4
        outpath = "../output/test_output/"
        clustered_solution = line_graph_clustering.cluster_for_k(k, line_graph, differences, False, 0)
        expected_solution = {'284': ['284'], '826': ['826', '820'], '986': ['986']}
        assert clustered_solution == expected_solution, (f"clustered_solution should be {expected_solution}, "
                                                         f"but is {clustered_solution}")
        pass
