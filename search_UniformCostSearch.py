# Mark Trinquero
# Basic Search Algorithm Examples
# Uniform-cost search

from __future__ import division
import random
import matplotlib.pyplot as plt
import pickle
import sys
import math
import networkx
import heapq
sys.path.insert(1,'lib')



# helper function to get total weight/cost of path
def get_path_cost(graph, path):
    if not path:
        return 0
    cost = 0
    for i in range(0, len(path) -1):
        edge_weight = graph.get_edge_data(path[i], path[i+1])['weight']
        cost += edge_weight
    return cost


def uniform_cost_search(graph, start, goal):
    if start == goal:
        path = []
        return path 
    node = start
    cost = int(0)
    tup = (cost, node)
    frontier = PriorityQueue() ## see priority queue class setup
    frontier.append(tup)
    explored = []
    
    while frontier:
        if frontier.size() == 0:
            print "ERROR EMPTY FRONTIER"
        cur_path = frontier.pop()
        path = cur_path[-1]
        node = cur_path[-1][-1]

        if node == goal:
            return path
        
        explored.append(node)
        neighbors = graph.neighbors(node)

        
        for adjacent in neighbors:
            if adjacent not in explored: 
                new_path = list(path)
                new_path.append(adjacent)
                temp_cost = get_path_cost(graph, new_path)
                if adjacent not in frontier:
                    temp_tup = (temp_cost, new_path)
                    frontier.append(temp_tup)
                elif adjacent in frontier:
                    if temp_cost < cost:
                        old = (cost,)
                        frontier.remove(old)
                        temp_tup = (temp_cost, new_path)
                        frontier.append(temp_tup)                        


