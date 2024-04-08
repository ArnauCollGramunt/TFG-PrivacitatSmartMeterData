import pandas as pd
import numpy as np
import sys
from collections import deque

INFINITY = float("inf")

k = int(sys.argv[1])
data = pd.read_csv(sys.argv[2], header=None)

data['row_id'] = range(len(data))

def OptimalUnivariantMicroaggregation():
    new_df = pd.DataFrame()
    for column in range(len(data.columns) - 1):
        partition = data.iloc[:, [column, -1]]
        partition = partition.sort_values(by=column)
        init_graph = []
        nodes = ["g0"] + partition.iloc[:, 0].to_list()
        original_values = partition.iloc[:, 1].to_list()
        for i in range(len(nodes)):
            for j in range(i + k, min(i + 2 * k, len(nodes))):
                cluster_values = nodes[i + 1 : j + 1]
                cluster_values = [float(value) for value in cluster_values]
                mean = np.mean(cluster_values)
                sse = np.sum((cluster_values - mean) ** 2)
                init_graph.append([f"g{i}",f"g{j}",sse])
        graph = Graph(init_graph)
        shortest_path, cost = graph.shortest_path("g0",f"g{len(nodes)-1}") 
        new_values = []
        while len(new_values) < data.shape[0]:
            new_values.insert(0, None)
        follow = shortest_path[1]
        actual = "g0"
        for ind in range(1,len(shortest_path)):
            follow = shortest_path[ind]
            upper_lim = int(''.join(filter(str.isdigit, follow)))
            lower_lim = int(''.join(filter(str.isdigit, actual))) + 1
            cont = 0
            div = 0
            for i in range(lower_lim,upper_lim+1):
                cont = nodes[i] + cont
                div = div + 1
            mean = cont / div
            for i in range(lower_lim,upper_lim+1):
                new_values[original_values[i-1]] = mean
            actual = follow
        new_df[column] = new_values

    return new_df


class Graph:
    def __init__(self, edges):
        """Reads graph definition and stores it. Each line of the graph
        definition file defines an edge by specifying the start node,
        end node, and distance, delimited by spaces.

        Stores the graph definition in two properties which are used by
        Dijkstra's algorithm in the shortest_path method:
        self.nodes = set of all unique nodes in the graph
        self.adjacency_list = dict that maps each node to an unordered set of
        (neighbor, distance) tuples.
        """

        # Read the graph definition file and store in graph_edges as a list of
        # lists of [from_node, to_node, distance]. This data structure is not
        # used by Dijkstra's algorithm, it's just an intermediate step in the
        # create of self.nodes and self.adjacency_list.
        graph_edges = []

        for i in range(len(edges)):
            edge_from, edge_to, cost, = edges[i][0], edges[i][1], edges[i][2]
            graph_edges.append((edge_from, edge_to, float(cost)))

        self.nodes = set()
        for edge in graph_edges:
            self.nodes.update([edge[0], edge[1]])

        self.adjacency_list = {node: set() for node in self.nodes}
        for edge in graph_edges:
            self.adjacency_list[edge[0]].add((edge[1], edge[2]))

    def shortest_path(self, start_node, end_node):
        """Uses Dijkstra's algorithm to determine the shortest path from
        start_node to end_node. Returns (path, distance).
        """

        unvisited_nodes = self.nodes.copy()  # All nodes are initially unvisited.

        # Create a dictionary of each node's distance from start_node. We will
        # update each node's distance whenever we find a shorter path.
        distance_from_start = {
            node: (0 if node == start_node else INFINITY) for node in self.nodes
        }

        # Initialize previous_node, the dictionary that maps each node to the
        # node it was visited from when the the shortest path to it was found.
        previous_node = {node: None for node in self.nodes}

        while unvisited_nodes:
            # Set current_node to the unvisited node with shortest distance
            # calculated so far.
            current_node = min(
                unvisited_nodes, key=lambda node: distance_from_start[node]
            )
            unvisited_nodes.remove(current_node)

            # If current_node's distance is INFINITY, the remaining unvisited
            # nodes are not connected to start_node, so we're done.
            if distance_from_start[current_node] == INFINITY:
                break

            # For each neighbor of current_node, check whether the total distance
            # to the neighbor via current_node is shorter than the distance we
            # currently have for that node. If it is, update the neighbor's values
            # for distance_from_start and previous_node.
            for neighbor, distance in self.adjacency_list[current_node]:
                new_path = distance_from_start[current_node] + distance
                if new_path < distance_from_start[neighbor]:
                    distance_from_start[neighbor] = new_path
                    previous_node[neighbor] = current_node

            if current_node == end_node:
                break # we've visited the destination node, so we're done

        # To build the path to be returned, we iterate through the nodes from
        # end_node back to start_node. Note the use of a deque, which can
        # appendleft with O(1) performance.
        path = deque()
        current_node = end_node
        while previous_node[current_node] is not None:
            path.appendleft(current_node)
            current_node = previous_node[current_node]
        path.appendleft(start_node)

        return path, distance_from_start[end_node]


example_df = OptimalUnivariantMicroaggregation()

example_df.to_csv("maskedDataOUM.csv", index=False, header=False)