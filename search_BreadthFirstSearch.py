# Mark Trinquero
# Basic Search Algorithm Examples
# Breath First Search


from __future__ import division
import random
import matplotlib.pyplot as plt
import pickle
import sys
import math
import networkx
import heapq
sys.path.insert(1,'lib')


def breadth_first_search(graph, start, goal):
    if start == goal:
        path = []
        return path    
    queue = []
    queue.append([start])   #push the first path into the queue
    while queue:
        path = queue.pop(0)  # get the first path from the queue
        node = path[-1] # get the last node from the path
        if node == goal:  # path found
            return path
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        neighbors = graph.neighbors(node)

        for adjacent in neighbors:
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)
            if adjacent == goal:    # if goal is reached, return that path
                return new_path
        # return path


def draw_graph(graph, node_positions={}, start=None, goal=None, path=[]):
    explored = list(graph.get_explored_nodes())
    labels ={}
    for node in graph:
        labels[node]=node
    if not node_positions:
        node_positions = networkx.spring_layout(graph)

    networkx.draw_networkx_nodes(graph, node_positions)
    networkx.draw_networkx_edges(graph, node_positions, style='dashed')
    networkx.draw_networkx_labels(graph,node_positions, labels)
    networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored, node_color='g') 

    if path:
        edges = [(path[i], path[i+1]) for i in range(0, len(path)-1)]
        networkx.draw_networkx_edges(graph, node_positions, edgelist=edges, edge_color='b')
    if start:
        networkx.draw_networkx_nodes(graph, node_positions, nodelist=[start], node_color='b')
    if goal:
        networkx.draw_networkx_nodes(graph, node_positions, nodelist=[goal], node_color='y')

    plt.plot()
    plt.show()


