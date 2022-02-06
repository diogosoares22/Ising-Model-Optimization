import numpy as np

from algorithms import metropolis_algorithm

if __name__ == "__main__":
    graph = np.load("contest_data/A.npy")
    ground_truth = np.load("contest_data/A_vec.npy")
    a = 41.27
    b = 1.79
    N = len(ground_truth)

    import time

    start = time.time()

    X, energies, overlaps = metropolis_algorithm(graph, ground_truth, a, b, 9000, 1, lambda x: x + 0.1, debug=True, branching_factor=2)

    end = time.time()
    delta = end - start

    print("Metropolis took {}".format(delta))

    np.save("contest_data/MarkovPolo", X)





