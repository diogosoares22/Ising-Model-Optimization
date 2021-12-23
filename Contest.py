import numpy as np

from functions import xvariant, compute_overlap, Functions, compute_local_overlap, get_random_difference_node, bfs
from generate_graph import generate_vector

def metropolis_algorithm(graph, a, b, steps, beta, beta_update_func, X=None, seed=None, debug=True, branching_factor=5):
    """ Standard metropolis algorithm  with uniform random walk """
    if seed != None:
        np.random.seed(seed)
    
    N = graph.shape[0]

    if branching_factor is None:
        branching_factor = N

    funcs = Functions(graph, a, b, N)
    
    if X is None:
        X = generate_vector(N)
    best_X = X 
    lowest_energy = energy = funcs.hamiltonian_of_gibbs_model_vectorized(X)

    energies = [energy]

    for i in range(steps):
            
        X_variants = [xvariant(X, j) for j in np.random.choice(range(N), branching_factor)]
        X_variants_acceptance_prob = [funcs.compute_acceptance_probability(beta, X, x_variant) * (1/branching_factor) for x_variant in X_variants]
        X_variants_acceptance_prob = [1 - sum(X_variants_acceptance_prob)] + X_variants_acceptance_prob
        X_variants = [X] + X_variants

        X = X_variants[np.random.choice(range(len(X_variants)), p = X_variants_acceptance_prob)]

        energy = funcs.hamiltonian_of_gibbs_model_vectorized(X)

        energies.append(energy)

        if (energy < lowest_energy):
            if (debug):
                print("At step {}, we found a better energy {}".format(i, energy))
            lowest_energy = energy
            best_X = X
        
        beta = beta_update_func(beta)
    return best_X, energies


if __name__ == "__main__":
    graph = np.load("A_test.npy")    
    a = 5
    b = 1
    #best_X = np.load("A_test.npy")
    best_X, energies = metropolis_algorithm(graph, a, b, 100, 1, lambda x: x + 0.1, debug=True, branching_factor=5)
    print(best_X.shape)
    np.save('./submission/MarkovPolo.npy', best_X)