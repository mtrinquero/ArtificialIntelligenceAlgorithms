# Mark Trinquero
# Basic Search Algorithm Examples 
# Priority Queue


from __future__ import division
import random
import matplotlib.pyplot as plt
import pickle
import sys
import math
import networkx
import heapq
sys.path.insert(1,'lib')


class PriorityQueue():
    def __init__(self):
        self.queue = []
        self.current = 0    

    def next(self):
        if self.current >=len(self.queue):
            self.current
            raise StopIteration
        out = self.queue[self.current]
        self.current += 1
        return out

    def pop(self):
        # https://docs.python.org/2/library/heapq.html
        smallest = heapq.heappop(self.queue)
        return smallest
        
    def remove(self, nodeId):
        #http://stackoverflow.com/questions/10162679/python-delete-element-from-heap
        self.queue[nodeId] = self.queue[-1]
        self.queue.pop()
        heapq.heapify(self.queue)
        return self

    def __iter__(self):
        return self

    def __str__(self):
        return 'PQ:[%s]'%(', '.join([str(i) for i in self.queue]))

    def append(self, node):
        heapq.heappush(self.queue, node)
        return self
    
    def __contains__(self, key):
        self.current = 0
        return key in [n for v,n in self.queue]

    def __eq__(self, other):
        return self == other

    def size(self):
        return len(self.queue)
    
    def clear(self):
        self.queue = []
        
    def top(self):
        return self.queue[0]

    __next__ = next
