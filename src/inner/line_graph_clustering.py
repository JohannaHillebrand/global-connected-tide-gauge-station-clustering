import time

import networkx
from loguru import logger


def select_centers(g: networkx.Graph, possible_centers_dict: dict, node_belonging_dict: dict):
    """
    Select centers for the clustering problem, based on the possible centers that were computed before
    :param node_belonging_dict:
    :param g: graph
    :param possible_centers_dict: dictionary containing for each node a graph containing the nodes that could be in
    the cluster if the
    node was the center
    :return:
    """
    working_graph = g.copy()
    current_centers = {}
    # find node that has only one neighbor as starting point
    while len(working_graph.nodes()) > 0:
        start_node = None
        for node in working_graph.nodes:
            if len(list(working_graph.edges(node))) <= 1:
                start_node = node
                break
        if start_node is None:
            if not networkx.is_forest(working_graph):
                logger.info(f"There is a cycle in the input graph, please remove and try again")
                exit(1)
        current_centers, possible_centers_dict, node_belonging_dict, working_graph = find_largest_cluster(
            current_centers,
            node_belonging_dict,
            possible_centers_dict,
            start_node, working_graph)

    return current_centers


def find_largest_cluster(current_centers: dict, node_belonging_dict: dict, possible_centers_dict: dict, start_node: int,
                         working_graph: networkx.Graph):
    """
    Find the largest cluster for a given start node
    :param current_centers: a dictionary containing every prior indentified center, and a graph containing the nodes
    that are in the associated cluster
    :param node_belonging_dict:
    :param possible_centers_dict:
    :param start_node:
    :param working_graph:
    :return:
    """
    # these clusters contain the start_node
    center_option_for_start_node = node_belonging_dict[start_node]
    # find largest cluster
    longest_length = 0
    center = start_node
    if len(center_option_for_start_node) > 1:
        # only if node belongs to more clusters than its own (otherwise node it is its own center)
        for node in center_option_for_start_node:
            if node not in possible_centers_dict:
                continue
            if len(possible_centers_dict[node].nodes) > longest_length:
                longest_length = len(possible_centers_dict[node].nodes)
                center = node
    # add center to list of centers
    current_centers[center] = possible_centers_dict[center]
    # remove center from graph and all associated nodes from graph, node_belonging_dict and possible_centers_dict
    working_graph.remove_nodes_from(possible_centers_dict[center].nodes)
    new_possible_centers_dict = {}
    for current_key in possible_centers_dict:
        # remove nodes that have been assigned to this center from all other possible clusters, so they do not
        # disturb the size, when calculating the next center point
        new_possible_centers_dict[current_key] = possible_centers_dict[current_key].copy()
        new_possible_centers_dict[current_key].remove_nodes_from(possible_centers_dict[center].nodes)
    for node in possible_centers_dict[center].nodes:
        if node in node_belonging_dict:
            node_belonging_dict.pop(node)
        if node in new_possible_centers_dict:
            new_possible_centers_dict.pop(node)

    return current_centers, new_possible_centers_dict, node_belonging_dict, working_graph


def check_distances_for_subgraph(current_radius: float, distance_dict: dict, center_node: int, next_node: int,
                                 g: networkx.Graph):
    """
    Compute for a linestring which nodes can be added to the cluster, such that they are connected, and are not
    further away from the center than the radius. If the distance from next node on the line to the center is more
    than the radius the function will return the previous nodes
    :param g:
    :param next_node:
    :param distance_dict:
    :param current_radius:
    :param center_node:
    :return:
    """
    working_graph = g.copy()
    # subgraph containing all nodes that would be in the connected cluster is this node was the center
    subgraph = networkx.Graph()
    subgraph.add_node(center_node)
    old_node = center_node
    added_nodes = True

    while added_nodes:
        # The second part is only necessary for the binary search, as inf is used for the difference between two nodes
        # that do not have sufficient overlap, but are still connected in the graph, but we do not want those two to
        # end up in a cluster
        if (distance_dict[center_node][next_node] <= current_radius) and (
                distance_dict[center_node][next_node] < float("inf")):
            subgraph.add_edge(old_node, next_node)
            working_graph.remove_node(old_node)
            old_node = next_node
            # check if next node exists !
            try:
                next_node = list(working_graph.edges(old_node))[0][1]
            except IndexError:
                break
            added_nodes = True
        else:
            added_nodes = False
    return subgraph


def check_distances_for_subgraph_with_threshold(current_radius: float, distance_dict: dict, center_node: int,
                                                next_node: int, g: networkx.Graph, current_threshold: int):
    """
    Compute for a linestring which nodes can be added to the cluster, such that they are connected, and are not
    further away from the center than the radius. If the distance from next node on the line to the center is more
    than the radius the function will return the previous nodes
    Consider the threshold for the overlap between the center and the nodes in the cluster
    :param current_threshold:
    :param g:
    :param next_node:
    :param distance_dict:
    :param current_radius:
    :param center_node:
    :return:
    """
    working_graph = g.copy()
    # subgraph containing all nodes that would be in the connected cluster is this node was the center
    subgraph = networkx.Graph()
    subgraph.add_node(center_node)
    old_node = center_node
    added_nodes = True

    while added_nodes:
        if (distance_dict[center_node][next_node][1] <= current_radius) and (
                distance_dict[center_node][next_node][0] >= current_threshold):
            subgraph.add_edge(old_node, next_node)
            working_graph.remove_node(old_node)
            old_node = next_node
            # check if next node exists !
            try:
                next_node = list(working_graph.edges(old_node))[0][1]
            except IndexError:
                break
            added_nodes = True
        else:
            added_nodes = False
    return subgraph


def compute_cluster_for_each_node(g: networkx.Graph, current_radius: float, distance_dict: dict, center_node: int,
                                  which_node_belongs_to_which_centers: dict):
    """
    Compute the subgraph for each node, that would be in the connected cluster, if this node was the center
    :param which_node_belongs_to_which_centers:
    :param center_node:
    :param g:
    :param current_radius:
    :param distance_dict:
    :return:
    """
    subgraph = networkx.Graph()
    # check if center node has two neighbors, one or none and proceed to compute the subgraph containing all nodes
    # that are within the radius
    if len(list(g.edges(center_node))) > 1:
        left_node = list(g.edges(center_node))[0][1]
        right_node = list(g.edges(center_node))[1][1]
        # check which nodes to add on the left side
        left_subgraph = check_distances_for_subgraph(current_radius, distance_dict, center_node, left_node, g)
        # check which nodes to add on the right side
        right_subgraph = check_distances_for_subgraph(current_radius, distance_dict, center_node, right_node, g)
        left_subgraph.add_edges_from(right_subgraph.edges)
        subgraph = left_subgraph
    elif len(list(g.edges(center_node))) == 1:
        next_node = list(g.edges(center_node))[0][1]
        subgraph = check_distances_for_subgraph(current_radius, distance_dict, center_node, next_node, g)
    elif len(list(g.edges(center_node))) == 0:
        subgraph.add_node(center_node)
    # add to node_belonging dictionary to save for each node to which subgraph it belongs
    for node in subgraph.nodes:
        if node in which_node_belongs_to_which_centers:
            which_node_belongs_to_which_centers[node].append(center_node)
        else:
            which_node_belongs_to_which_centers[node] = [center_node]

    return subgraph, which_node_belongs_to_which_centers


def compute_cluster_for_each_node_demanding_overlap(graph: networkx.Graph, current_radius: float, distance_dict: dict,
                                                    center_node: int,
                                                    which_node_belongs_to_which_centers: dict, overlap: int):
    """
    Compute the subgraph for each node, that would be in the connected cluster, if this node was the center. Here we
    also know how large the gaps in the timeseries between the nodes are, and we demand that the center has an
    overlap of a certain percentage with each node in its cluster
    :param graph:
    :param current_radius:
    :param distance_dict:
    :param center_node:
    :param which_node_belongs_to_which_centers:
    :param overlap:
    :return:
    """
    g = graph.copy()
    subgraph = networkx.Graph()

    # check if center node has two neighbors, one or none and proceed to compute the subgraph containing all nodes
    # that are within the radius
    if len(list(g.edges(center_node))) > 1:
        left_node = list(g.edges(center_node))[0][1]
        right_node = list(g.edges(center_node))[1][1]
        # check which nodes to add on the left side
        left_subgraph = check_distances_for_subgraph_with_threshold(current_radius, distance_dict, center_node,
                                                                    left_node, g, overlap)
        # check which nodes to add on the right side
        right_subgraph = check_distances_for_subgraph_with_threshold(current_radius, distance_dict, center_node,
                                                                     right_node, g, overlap)
        subgraph.add_edges_from(right_subgraph.edges)
        subgraph.add_nodes_from(right_subgraph.nodes)
        subgraph.add_edges_from(left_subgraph.edges)
        subgraph.add_nodes_from(left_subgraph.nodes)
    elif len(list(g.edges(center_node))) == 1:
        next_node = list(g.edges(center_node))[0][1]
        subgraph = check_distances_for_subgraph_with_threshold(current_radius, distance_dict, center_node, next_node, g,
                                                               overlap)
    elif len(list(g.edges(center_node))) == 0:
        subgraph.add_node(center_node)
    # add to node_belonging dictionary to save for each node to which subgraph it belongs
    for node in subgraph.nodes:
        if node in which_node_belongs_to_which_centers:
            which_node_belongs_to_which_centers[node].append(center_node)
        else:
            which_node_belongs_to_which_centers[node] = [center_node]
    return subgraph, which_node_belongs_to_which_centers


def divide_graph_into_connected_components(g: networkx.Graph):
    """
    Divide a graph into connected graphs
    :param g: graph
    :return:
    """
    connected_components = []
    for current_graph in networkx.connected_components(g):
        connected_components.append(g.subgraph(current_graph))
    return connected_components


def compute_clustering_for_given_radius(g: networkx.Graph, current_radius: float, distance_dict: dict,
                                        is_threshold_wanted: bool, current_threshold: float):
    """
    Compute the clustering for a given radius
    :param is_threshold_wanted:
    :param current_threshold:
    :param distance_dict:
    :param g: graph
    :param current_radius: radius of the clusters
    :return:
    """
    # logger.info(f"Start clustering for radius: {current_radius}")
    # divide graph into connected components
    connected_components = divide_graph_into_connected_components(g)
    # for each connected subgraph compute the possible cluster for each node
    node_belonging_dict = {}
    possible_centers_dict = {}
    all_center_nodes = {}
    for component in connected_components:
        # compute the largest cluster for each node, if that node would be the center
        for node in component.nodes:
            if is_threshold_wanted:
                cluster, node_belonging_dict = compute_cluster_for_each_node_demanding_overlap(component.copy(),
                                                                                               current_radius,
                                                                                               distance_dict,
                                                                                               node,
                                                                                               node_belonging_dict,
                                                                                               current_threshold)
            else:
                cluster, node_belonging_dict = compute_cluster_for_each_node(component.copy(), current_radius,
                                                                             distance_dict,
                                                                             node,
                                                                             node_belonging_dict)
            possible_centers_dict[node] = cluster
        # select centers for each subgraph
        current_center_nodes = select_centers(component, possible_centers_dict, node_belonging_dict)
        all_center_nodes.update(current_center_nodes)
    # add up number of centers
    current_number_of_centers = len(all_center_nodes.keys())
    return current_number_of_centers, all_center_nodes


def binary_search(g: networkx.Graph, k: int, distance_dict: dict, threshold_wanted: bool, overlap: float):
    """
    Binary search to find the radius of the clusters
    :param overlap:
    :param threshold_wanted:
    :param distance_dict:
    :param k: number of clusters
    :param g: graph
    :return: radius of the clusters, clustering
    """
    # logger.info("Start binary search")
    search_center_nodes = {}
    # compute maximal possible distance between two nodes and compute set containing all possible distances
    all_distances = set()
    for first_node in distance_dict.keys():
        for second_node in distance_dict[first_node].keys():
            all_distances.add(distance_dict[first_node][second_node])
    # sort all possible distance values
    all_distances_sorted = sorted(all_distances)
    # start binary search
    array_length = len(all_distances_sorted)
    left_index = 0
    right_index = array_length - 1
    current_distance = None
    current_number_of_centers = None
    count_number_of_iterations = 0
    closest_centers = float('inf')  # Initialize with infinity
    closest_radius = None
    smallest_radius = float('inf')
    while left_index <= right_index:
        count_number_of_iterations += 1
        # this is the distance that will be tested
        mid_index = (left_index + right_index) // 2
        mid = all_distances_sorted[mid_index]
        temp_number_of_centers, current_center_nodes = compute_clustering_for_given_radius(g, mid, distance_dict,
                                                                                           threshold_wanted, overlap)
        if temp_number_of_centers <= k:
            right_index = mid_index - 1
            # Update the closest match if this is closer to k than previous attempts
            if abs(k - temp_number_of_centers) < abs(k - closest_centers):
                closest_centers = temp_number_of_centers
                closest_radius = mid
                closest_center_nodes = current_center_nodes
            if temp_number_of_centers == k:
                # if number of centers is already k, check if it is the solution with the smallest radius,
                # if this is the
                # case save the last solution for which the number of centers was k
                if mid < smallest_radius:
                    smallest_radius = mid
                    current_distance = mid
                    current_number_of_centers = temp_number_of_centers
                    search_center_nodes = current_center_nodes
        else:
            left_index = mid_index + 1
            # Update the closest match if this is closer to k than previous attempts
            if abs(k - temp_number_of_centers) < abs(k - closest_centers):
                closest_centers = temp_number_of_centers
                closest_radius = mid
                closest_center_nodes = current_center_nodes

    if current_distance is None and closest_radius is not None:
        logger.warning(
            f"No solution found for this k = {k}, the closest solution is returned with {closest_centers} centers")
        current_distance = closest_radius
        current_number_of_centers = closest_centers
        search_center_nodes = closest_center_nodes

    return current_distance, search_center_nodes, current_number_of_centers, count_number_of_iterations


def cluster_for_k(k, current_graph, current_distances, current_threshold_wanted, current_threshold):
    """
    Cluster for a given k, if the clustering for the given k is not possible, the closest solution is returned,
    if this is not wanted, it needs to be handled by the user.
    :param current_threshold: 
    :param current_threshold_wanted: 
    :param current_distances: 
    :param k:
    :param current_graph:
    :return:
    """

    start_time = time.time()
    max_radius, center_dict, number_centers, number_iterations = binary_search(current_graph, k, current_distances,
                                                                               current_threshold_wanted,
                                                                               current_threshold)
    if max_radius is None:
        return None
    end_time = time.time()
    elapsed_time = end_time - start_time
    final_dict = {}
    for key in center_dict.keys():
        final_dict[key] = list(center_dict[key].nodes)
    return final_dict, max_radius, elapsed_time, number_iterations
