from algorithms import metropolis_algorithm, houdayer_algorithm
from generate_graph import generate_graph

import matplotlib.pyplot as plt
import numpy as np
import time


#if __name__ == "__main__":
t_0 = time.process_time()
N = 100
d = 3
a = 4 #3<a<6
b = 2*d-a
steps = 15 # take a multiple of 5
nbr_experiment =1
beta_update_func = lambda x : x + 0.01
initial_beta = 0.01

x_step = np.arange(steps+1)
y_avgOverlap = np.zeros((3,steps+1))

ground_truth, graph = generate_graph(N, a, b)
print("Graph generated")

for k in range(1,nbr_experiment+1):
    t_ini = time.process_time()
    _ ,_ , metropolis_overlaps = metropolis_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, debug=False)
    _, _, _, _, houdayer_overlaps = houdayer_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, n0=1, debug=False)
    _, _, _, _, mixed_overlaps = houdayer_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, n0=5, debug=False)
    t_final = time.process_time()
    y_avgOverlap[0] += metropolis_overlaps
    y_avgOverlap[1] += houdayer_overlaps
    print(len(mixed_overlaps))
    y_avgOverlap[2] += mixed_overlaps
    delta_t = t_final-t_ini
    if k%5==0 :
        print("Experiment %d done in %d s \n Total time : %d s" % (k,delta_t,t_final-t_0))
y_avgOverlap/=nbr_experiment
plt.plot(x_step,y_avgOverlap[0], label='Metropolis algorithm')
plt.plot(x_step,y_avgOverlap[1], label='Houdayer algorithm')
plt.plot(x_step,y_avgOverlap[2], label='Mixed')
plt.legend()
plt.title("Average over " + str(nbr_experiment) + " experiments")
plt.xlabel("Number of step")
plt.ylabel("Average overlap")
plt.show()
