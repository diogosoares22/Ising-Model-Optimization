import numpy as np
from np_cache import np_cache
import random


class Functions:

    def __init__(self, graph, a, b, N):
        self.graph = graph
        self.a = a
        self.b = b
        self.N = N
        self.precomputed_division_log = np.log(self.a/ self.b)
        self.precomputed_division_with_minus_log = np.log((1 - self.a/N)/(1 - self.b/N))

    @np_cache()
    def hamiltonian_of_gibbs_model(self, x):
        """ compute the energy of the hamiltonian for the gibbs model """
        current_sum = 0
        N = len(x)
        for i in range(N):
            for j in range(i+1, N):
                is_edge = self.graph[i,j]
                mult = x[i]*x[j]
                hij = 1/2 * (is_edge * self.precomputed_division_log + (1 - is_edge) * self.precomputed_division_with_minus_log) 
                current_sum -= hij * mult
        return current_sum

    def compute_acceptance_probability(self, beta, current_state, future_state):
        """ compute the acceptance probability for two states given a cost function """
        return min(1, np.exp(-beta * (self.hamiltonian_of_gibbs_model(future_state) - self.hamiltonian_of_gibbs_model(current_state))))

def compute_overlap(x_star, x_pred):
    """ compute the overlap between x and x_star """
    return abs(np.dot(x_star, x_pred)) / len(x_star)

def compute_local_overlap(x1, x2):
    """ compute the local overlap between x1 and x2 """
    return x1 * x2

def get_random_difference_node(y):
    """ outputs the index of a random occurrence where x1i does not match x2i """
    numbers = []
    for i in range(len(y)):
        if y[i] == -1:
            numbers.append(i)
    return random.choice(numbers)

def bfs(visited, graph, node):
    """ breadth first search for a given node """
    visited = []   # List to keep track of visited nodes.
    queue = []     # Initialize a queue
    
    visited.append(node)
    queue.append(node)

    while queue:
        s = queue.pop(0) 

        for possible_neighbour in range(len(graph[s])):
            neighbour = graph[s][possible_neighbour]
            if neighbour == 1 and possible_neighbour not in visited:
                visited.append(possible_neighbour)
                queue.append(possible_neighbour)
    return visited



def xvariant(x, index):
    """ returns a variant of x with a certain index flipped """
    x_variant = x.copy()
    x_variant[index] *= -1
    return x_variant