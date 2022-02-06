import numpy as np
import random

def decision(probability, seed=None):
    """ function to generate decision based on probability """
    if seed:
        random.seed(seed)
    return random.random() < probability

def generate_vector(N, seed=None):
    """ returns a vector with components uniformly distributed in {-1, 1} """
    if seed:
        np.random.seed(seed)
    return np.random.choice([-1, 1], size=(N))

def generate_graph(N, a, b, seed=None):
    """ returns a graph with undirected edges in matrix format, following some probability a and b normalized """
    vector = generate_vector(N, seed=seed)
    matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            mul = vector[i] * vector[j]
            if (mul == 1):
                isEdge = decision(a/N)
            else:
                isEdge = decision(b/N)
            matrix[i,j] = isEdge
            matrix[j,i] = isEdge
    return vector, matrix

if __name__ == "__main__":
    vec, matrix = generate_graph(1000, 200, 100, seed=25)