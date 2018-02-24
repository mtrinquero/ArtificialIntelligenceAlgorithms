# Mark Trinquero
# Basic Search Algorithm Examples
# Bi-Directional Uniform Cost Search

from __future__ import division
import random
import matplotlib.pyplot as plt
import pickle
import sys
import math
import networkx
import heapq
from osm2networkx import *
sys.path.insert(1,'lib')



def bidirectional_ucs(graph, start, goal):
    if start == goal:
        path = []
        return path
        node = start
    cost = int(0)
    tup = (cost, node)
    frontier = PriorityQueue()
    frontier.append(tup)
    explored = []
    node2 = goal
    cost2 = int(0)
    tup2 = (cost2, node2)
    frontier2 = PriorityQueue()
    frontier2.append(tup2)
    explored2 = []

    while frontier and frontier2:
        if frontier.size() != 0:
            cur_path = frontier.pop()
            path = cur_path[-1]
            node = cur_path[-1][-1]
            if node == goal or node in frontier2:
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
        
        if frontier2.size() != 0:
            cur_path2 = frontier2.pop()
            path2 = cur_path2[-1]
            node2 = cur_path2[-1][-1]
            if node2 == start or node2 in frontier:
                return path2
            explored2.append(node2)
            neighbors2 = graph.neighbors(node2)

        for adjacent in neighbors2:
            if adjacent not in explored2: 
                new_path2 = list(path2)
                new_path2.append(adjacent)
                temp_cost2 = get_path_cost(graph, new_path2)
                if adjacent not in frontier2:
                    temp_tup2 = (temp_cost2, new_path2)
                    frontier2.append(temp_tup2)
                elif adjacent in frontier2:
                    if temp_cost2 < cost2:
                        old2 = (cost2,)
                        frontier2.remove(old2)
                        temp_tup2 = (temp_cost2, new_path2)
                        frontier2.append(temp_tup2)                        


