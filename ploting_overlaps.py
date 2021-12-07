from algorithms import metropolis_algorithm, houdayer_algorithm
from generate_graph import generate_graph

def create_a_b_combination(d, r_values):
    combinations = []
    for r in r_values:
        a = (2*d) / (1 + r)
        b = a * r
        combinations.append((b, a))
    return combinations

def plot_results(model_name, r_values, results):
    import matplotlib.pyplot as plt
    x = r_values
    y = [results[r] for r in r_values]
    plt.plot(x, y)
    plt.ylabel("Limiting overlap")
    plt.xlabel("r value")
    plt.title(model_name + " limiting overlap in function of r")
    plt.show()

def plot_results_combined(models, r_values, results_for_model):
    import matplotlib.pyplot as plt
    for model in models:
        plt.plot(r_values, [results_for_model[model][r] for r in r_values], label=model)
    
    plt.ylabel("Limiting overlap")
    plt.xlabel("r value")
    plt.title("Multiple models limiting overlap in function of r")
    plt.legend()
    plt.show()


# let's consider N = 100, d = 3, let's consider 500 steps, and 100 iterations

if __name__ == "__main__":
    N = 100
    d = 3
    steps = 100
    iterations = 10
    beta_update_func = lambda x : x + 0.01
    initial_beta = 0.01
    r_values = [(i + 1) / 20 for i in range(8)]
    a_b_combination = create_a_b_combination(d, r_values)
    
    metropolis_results = {}
    houdayer_results = {}
    mixed_results = {}

    for r in r_values:
        metropolis_results[r] = []
        houdayer_results[r] = []
        mixed_results[r] = []
    for i in range(len(a_b_combination)): 

        for iteration in range(iterations):

            b = a_b_combination[i][0]
            a = a_b_combination[i][1]
        
            ground_truth, graph = generate_graph(N, a, b)

            _ ,_ , metropolis_overlaps = metropolis_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, debug=False)

            _, _, _, _, houdayer_overlaps = houdayer_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, n0=1, debug=False)

            _, _, _, _, mixed_overlaps = houdayer_algorithm(graph, ground_truth, a, b, steps, initial_beta, beta_update_func, n0=5, debug=False)
        
            metropolis_results[r_values[i]].append(metropolis_overlaps[-1])
            houdayer_results[r_values[i]].append(houdayer_overlaps[-1])
            mixed_results[r_values[i]].append(mixed_overlaps[-1])
            print("Iteration {}".format(iteration + 1))
        
        print("R value {}".format(r_values[i]))

        metropolis_results[r_values[i]] = sum(metropolis_results[r_values[i]]) / len(metropolis_results[r_values[i]])

        print("Metropolis Results: {}".format(metropolis_results[r_values[i]]))

        houdayer_results[r_values[i]] = sum(houdayer_results[r_values[i]]) / len(houdayer_results[r_values[i]])

        print("Houdayer Results: {}".format(houdayer_results[r_values[i]]))

        mixed_results[r_values[i]] = sum(mixed_results[r_values[i]]) / len(mixed_results[r_values[i]])

        print("Mixed Results: {}".format(mixed_results[r_values[i]]))

    
    plot_results("Metropolis Hastings", r_values, metropolis_results)
    plot_results("Houdayer", r_values, houdayer_results)
    plot_results("Mixed", r_values, mixed_results)
    plot_results_combined(["Metropolis", "Houdayer", "Mixed"], r_values, {"Metropolis" : metropolis_results, "Houdayer" : houdayer_results, "Mixed" : mixed_results})