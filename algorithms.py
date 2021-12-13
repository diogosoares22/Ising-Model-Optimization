import numpy as np

from generate_graph import generate_vector, generate_graph
from functions import xvariant, compute_overlap, Functions, compute_local_overlap, get_random_difference_node, bfs

def metropolis_algorithm(graph, ground_truth, a, b, steps, beta, beta_update_func, X=None, seed=None, debug=True):
    """ Standard metropolis algorithm  with uniform random walk """
    if seed != None:
        np.random.seed(seed)

    N = len(ground_truth)

    funcs = Functions(graph, a, b, N)
    
    if X is None:
        X = generate_vector(N)
    best_X = X 
    lowest_energy = energy = funcs.hamiltonian_of_gibbs_model(X)
    overlap = compute_overlap(ground_truth, X)

    overlaps = [overlap]
    energies = [energy]

    for i in range(steps):
            
        X_variants = [xvariant(X, j) for j in range(N)]
        X_variants_acceptance_prob = [funcs.compute_acceptance_probability(beta, X, x_variant) * (1/N) for x_variant in X_variants]
        X_variants_acceptance_prob = [1 - sum(X_variants_acceptance_prob)] + X_variants_acceptance_prob
        X_variants = [X] + X_variants

        X = X_variants[np.random.choice(range(len(X_variants)), p = X_variants_acceptance_prob)]

        energy = funcs.hamiltonian_of_gibbs_model(X)
        overlap = compute_overlap(ground_truth, X)

        energies.append(energy)
        overlaps.append(overlap)

        if (energy < lowest_energy):
            if (debug):
                print("At step {}, we found a better energy {}".format(i, energy))
            lowest_energy = energy
            best_X = X
        
        beta = beta_update_func(beta)

    if (debug):
        print("Ground_truth energy " + str(funcs.hamiltonian_of_gibbs_model(ground_truth)))
        print("Best energy found " + str(lowest_energy))
    return best_X, energies, overlaps

def houdayer_algorithm(graph, ground_truth, a, b, steps, beta, beta_update_func, n0=1, seed=None, debug=True):
    """ Houdayer Algorithm """
    
    if seed != None:
        np.random.seed(seed)

    N = len(ground_truth)

    funcs = Functions(graph, a, b, N)
    
    best_X1 = X1 = generate_vector(N)
    best_X2 = X2 = generate_vector(N)
    
    lowest_energy = energy = funcs.hamiltonian_of_gibbs_model(X1) + funcs.hamiltonian_of_gibbs_model(X2)
    
    energies = [energy]
    local_overlaps = [compute_local_overlap(X1, X2)]

    overlap = (compute_overlap(ground_truth, X1) + compute_overlap(ground_truth, X2)) / 2
    overlaps = [overlap]
    
    for i in range(steps // n0):
        
        # step 2

        ind = get_random_difference_node(local_overlaps[-1])

        energy = funcs.hamiltonian_of_gibbs_model(X1) + funcs.hamiltonian_of_gibbs_model(X2)

        cluster_nodes = bfs([], graph, ind, local_overlaps[-1])

        for node in cluster_nodes:
            X1[node] *= -1
            X2[node] *= -1

        energy = funcs.hamiltonian_of_gibbs_model(X1) + funcs.hamiltonian_of_gibbs_model(X2)

        # step 3

        X1, _, metropolis_overlaps_X1 = metropolis_algorithm(graph, ground_truth, a, b, n0, beta, beta_update_func, X=X1, debug=False)

        X2, _ , metropolis_overlaps_X2 = metropolis_algorithm(graph, ground_truth, a, b, n0, beta, beta_update_func, X=X2, debug=False)

        energy = funcs.hamiltonian_of_gibbs_model(X1) + funcs.hamiltonian_of_gibbs_model(X2)

        overlaps = overlaps + [(a + b) / 2 for a, b in zip(metropolis_overlaps_X1[1:], metropolis_overlaps_X2[1:])]

        # step 1
        
        local_overlap = compute_local_overlap(X1, X2)

        energies.append(energy)
        local_overlaps.append(local_overlap)

        if (energy < lowest_energy):
            if debug:
                print("At step {}, we found a better energy {}".format(i * n0, energy))
            lowest_energy = energy
            best_X1 = X1
            best_X2 = X2
        
        for i in range(n0):
            beta = beta_update_func(beta)

    return best_X1, best_X2, energies, local_overlaps, overlaps


        
if __name__ == "__main__":
    N = 200
    a = 2
    b = 1

    ground_truth, graph = generate_graph(N, a, b, seed=2)

    X1, _ , _ , _ = houdayer_algorithm(graph, ground_truth, a, b, 500, 0.1, lambda x : x + 0.01)  

    print(sum(local_overlaps[-1]))
