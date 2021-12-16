from algorithms import metropolis_algorithm, houdayer_algorithm
from generate_graph import generate_graph

import matplotlib.pyplot as plt
import numpy as np
import time

def update_overlap(algorithm, y_avgOverlap, graph, ground_truth, a, b, steps, initial_beta, beta_update_func):
    if algorithm=="Me" or algorithm=="Three":
        _ ,_ , metropolis_overlaps = metropolis_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, debug=False, branching_factor=5)
        y_avgOverlap[0] += metropolis_overlaps
    if algorithm=="H" or algorithm=="Three" :
            _, _, _, _, houdayer_overlaps = houdayer_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, n0=1, debug=False, branching_factor=5)
            y_avgOverlap[1] += houdayer_overlaps
    if algorithm =="Mi" or algorithm=="Three":
           _, _, _, _, mixed_overlaps = houdayer_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, n0=5, debug=False, branching_factor=5)
           y_avgOverlap[2] += mixed_overlaps

def plot_result_algorithm(algorithm, x_step, y_avgOverlap):
    if algorithm=="Me" or algorithm=="Three":
        plt.plot(x_step, y_avgOverlap[0], 'b', label='Metropolis algorithm')
    if algorithm=="H" or algorithm=="Three" :
        plt.plot(x_step, y_avgOverlap[1], 'r', label='Houdayer algorithm')
    if algorithm =="Mi" or algorithm=="Three":
        plt.plot(x_step ,y_avgOverlap[2], 'g', label='Mixed')
    plt.legend()
    plt.title("Average over " + str(nbr_experiment) + " experiments")
    plt.xlabel("Number of step")
    plt.ylabel("Average overlap")
    plt.show()    

#Choose the parameters of the study :
algorithm = "Three" #"Me"    for Metropolis
                    #"H"     for Houdayer
                    #"Mi"    for Mixed
                    #"Three" for using the three different algorithms
N = 100
d = 3
a = 5.5 #4.75<a<6 (to have r<r_c and b>0)
steps = 2000 # take a multiple of 5
nbr_experiment =100
beta_update_func = lambda x : x + 0.01
initial_beta = 0.01

#Define other parameters and launch the algorithm
b = 2*d-a
t_0 = time.process_time()
x_step = np.arange(steps+1)
y_avgOverlap = np.zeros((3,steps+1))
print("r = %f with this set up" %(b/a))
ground_truth, graph = generate_graph(N, a, b)
print("Graph generated")

for k in range(1,nbr_experiment+1):
    update_overlap(algorithm, y_avgOverlap, graph, ground_truth, a, b, steps, initial_beta, beta_update_func)
    t_final = time.process_time()
    if k%5==0 :
        print("Experiment %d done \nTotal time : %d min" % (k,(t_final-t_0)/60))
y_avgOverlap/=nbr_experiment
plot_result_algorithm(algorithm, x_step, y_avgOverlap)
