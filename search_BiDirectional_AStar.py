# Mark Trinquero
# Basic Search Algorithm Examples
# Bi-Directional A Star Search

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


def bidirectional_a_star(graph, start, goal, heuristic):
    if start == goal:
        path = []
        return path
    node = start
    a_cost = heuristic_euclid(graph, node, goal)
    cost = int(0) + a_cost
    tup = (cost, node)
    frontier = PriorityQueue()
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
                temp_cost = get_path_cost(graph, new_path) + heuristic_euclid(graph, adjacent, goal)
                if adjacent not in frontier:
                    temp_tup = (temp_cost, new_path)
                    frontier.append(temp_tup)
                elif adjacent in frontier:
                    if temp_cost < cost:
                        old = (cost,)
                        frontier.remove(old)
                        temp_tup = (temp_cost, new_path)
                        frontier.append(temp_tup)                        

