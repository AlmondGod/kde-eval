import numpy as np
from scipy.spatial import KDTree
import time
import matplotlib.pyplot as plt
from hbe.eLSH import ELSH

def student_kernel(x):
    return 1/((x)**10+1)

def adaptive_shell_algorithm(kernel_data, queries, k=1, num_spheres=50):
    print(f"adaptive shell algorithm start with k={k}")
    dataset_size, dimensions = kernel_data.shape
    gaussian_vectors = np.random.normal(loc=0, scale=1, size=(k, dimensions))
    projected_data = np.dot(kernel_data, gaussian_vectors.T) / np.sqrt(k)
    projected_queries = np.dot(queries, gaussian_vectors.T) / np.sqrt(k)
    print('init tree')
    tree = KDTree(projected_data)

    print(f"dataset size: {dataset_size}")
    def estimate_kernel_squared(query):
        distances, _ = tree.query(query, k=2)
        closest_distance = distances[0]
        print(f"closest distance: {closest_distance}")

        if closest_distance == 0.0:
            closest_distance = distances[1]
            print(f"Using second closest distance: {closest_distance}")
        
        kernel_sq_estimate = 0
        points_counted = 0
        current_radius = closest_distance
        total_points = 0
        iters = 0
        closest_radius = current_radius

        while total_points < dataset_size:
            count = tree.query_ball_point(query, current_radius, return_length=True)
            new_points = count - total_points
            print(f"closest edge {closest_radius}, far edge {current_radius}")
            if new_points > 0:
                kernel_sq_estimate += new_points * student_kernel(closest_radius)**2
                points_counted += new_points
            
            current_radius *= 2
            total_points = count
            if (iters > 0):
                closest_radius *= 2

            iters += 1

        return kernel_sq_estimate / points_counted if points_counted > 0 else 0

    epsilon = 0.5
    results = []

    for q in queries:
        query_start_time = time.time()
        variance = estimate_kernel_squared(projected_queries[np.where((queries == q).all(axis=1))[0][0]])
        
        j = 1
        r = np.random.randint(0, dataset_size-1)
        kernel_sumStudent = student_kernel(np.linalg.norm(kernel_data[r] - q))
        averageStudent = kernel_sumStudent/j
        t = 2 * variance / ((averageStudent)**2 * (epsilon)**2)

        while j < t:
            r = np.random.randint(0, dataset_size-1)
            kernel_sumStudent += student_kernel(np.linalg.norm(kernel_data[r] - q))
            j += 1
            averageStudent = kernel_sumStudent/j
            t = 2 * variance / (epsilon**2 * averageStudent**2)
        
        print("j: ", j)
                
        averageStudent = kernel_sumStudent/j
        query_time = time.time() - query_start_time
        print(query_time)
        actual_kernel = np.mean([student_kernel(np.linalg.norm(x - q)) for x in kernel_data])
        
        actual_kernel_sq = np.mean([student_kernel(np.linalg.norm(x - q)**2) for x in kernel_data])
        percent_error = np.abs(averageStudent - actual_kernel) / actual_kernel
        print(f"k, e {actual_kernel}, {percent_error}")
        
        print("estimatedvar over kernelsq:", variance / actual_kernel_sq)
        results.append((averageStudent, actual_kernel, percent_error, query_time))

    overall_error = np.mean([r[2] for r in results])
    execution_time = sum([r[3] for r in results])

    return results, overall_error, execution_time

def hashing_based_algorithm(kernel_data, queries):
    print("Hashing Based algorithm start")
    kde = GHBE(kernel_data.T, 
           tau=0.1 / np.sqrt(1000), eps=0.2, gamma=0.6,
           max_levels=8, max_tables_per_level=500,
           k_factor=2, w_factor=1.0)

    results = []
    for q in queries:
        query_start_time = time.time()
        est = kde.AMR(q)
        actual = np.mean([np.exp(-np.linalg.norm(q - x)**2) for x in kernel_data])
        percent_error = np.abs(est - actual) / actual
        query_time = time.time() - query_start_time
        results.append((est, actual, percent_error, query_time))

    overall_error = np.mean([r[2] for r in results])
    execution_time = sum([r[3] for r in results])

    return results, overall_error, execution_time

class GHBE:
    def __init__(self, X, tau=1e-3, eps=0.1, gamma=0.5, max_levels=10, max_tables_per_level=1000, k_factor=3, w_factor=np.sqrt(2.0/np.pi)):
        self.eps = eps
        self.R = np.sqrt(np.log(1 / tau))
        self.gamma = gamma
        self.I = min(int(np.ceil(np.log2(1 / tau))), max_levels)
        self.mui = np.array([(1 - self.gamma) ** i for i in range(self.I)])
        self.ti = np.sqrt(np.log(1/self.mui)) / 2.0
        self.ki = [int(k_factor * np.ceil(self.R * self.ti[j])) for j in range(self.I)]
        self.wi = self.ki / np.maximum(self.ti, 1) * w_factor
        self.RelVar = lambda mu: 1.0 / np.power(mu, 0.75)
        self.Mi = [min(int(np.ceil(eps**-2 * self.RelVar(self.mui[j]))), max_tables_per_level) for j in range(self.I)]
        
        print(f"Building {sum(self.Mi)} hash functions across {self.I} levels")
        self.HTA = [ELSH(self.Mi[j], X, self.wi[j], self.ki[j],
                         lambda x, y: np.exp(-np.linalg.norm(x-y)**2))
                    for j in range(self.I)]
    
    def AMR(self, q):
        for i in range(self.I):
            Z = 0.0
            for j in range(self.Mi[i]):
                Z += self.HTA[i].evalquery(q)
            if Z / (self.Mi[i]+0.0) >= self.mui[i]:
                return Z / (self.Mi[i])
        return Z / (self.Mi[i])

def compare_k_values(kernel_data, queries, num_trials=5):
    k_values = [1]
    adaptive_results = {}
    
    for k in k_values:
        trial_results = []
        for _ in range(num_trials):
            results, overall_error, execution_time = adaptive_shell_algorithm(kernel_data, queries, k=k)
            trial_results.append((overall_error, execution_time))
        
        avg_error = np.mean([r[0] for r in trial_results])
        avg_time = np.mean([r[1] for r in trial_results])
        adaptive_results[k] = (results, avg_error, avg_time)
    
    hashing_trial_results = []
    for _ in range(num_trials):
        hashing_results, hashing_overall_error, hashing_time = hashing_based_algorithm(kernel_data, queries)
        hashing_trial_results.append((hashing_overall_error, hashing_time))
    
    avg_hashing_error = np.mean([r[0] for r in hashing_trial_results])
    avg_hashing_time = np.mean([r[1] for r in hashing_trial_results])
    
    return adaptive_results, (hashing_results, avg_hashing_error, avg_hashing_time)
    

def plot_k_comparison(adaptive_results, hashing_results):
    k_values = list(adaptive_results.keys())
    
    errors = [result[1] for result in adaptive_results.values()]
    times = [result[2] for result in adaptive_results.values()]
    
    hashing_error = hashing_results[1]
    hashing_time = hashing_results[2]
    
    # errors
    plt.figure(figsize=(12, 6))
    plt.bar(k_values, errors, label='Adaptive Shell')
    plt.axhline(y=hashing_error, color='r', linestyle='--', label='Hashing Based')
    plt.xlabel('k value')
    plt.ylabel('Average Overall Error')
    plt.title('Error Comparison: Adaptive Shell (varying k) vs Hashing Based')
    plt.legend()
    plt.show()
    
    # execution Times
    plt.figure(figsize=(12, 6))
    plt.bar(k_values, times, label='Adaptive Shell')
    plt.axhline(y=hashing_time, color='r', linestyle='--', label='Hashing Based')
    plt.xlabel('k value')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Execution Time Comparison: Adaptive Shell (varying k) vs Hashing Based')
    plt.legend()
    plt.show()
    
    # error vs time scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(times, errors, label='Adaptive Shell')
    for k, time, error in zip(k_values, times, errors):
        plt.annotate(f'k={k}', (time, error))
    plt.scatter([hashing_time], [hashing_error], color='r', label='Hashing Based')
    plt.annotate('Hashing', (hashing_time, hashing_error))
    plt.xlabel('Average Execution Time (seconds)')
    plt.ylabel('Average Overall Error')
    plt.title('Error vs Execution Time: Adaptive Shell (varying k) vs Hashing Based')
    plt.legend()
    plt.show()

import gzip
with gzip.open('large_data/SUSY.csv.gz', 'rb') as f:
    susy_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)

# with gzip.open('large_data/HIGGS.csv.gz', 'rb') as f:
#     higgs_data = np.genfromtxt(f, delimiter=',', max_rows=4000000)
# home_data = np.genfromtxt('large_data/HT_Sensor_dataset.dat', skip_header=1, delimiter=None)
data = susy_data
kernel_data = data[0:1000]
queries = data[2000:2010]
# kernel_data = np.loadtxt('large_data/shuttle.tst')
# queries = np.loadtxt('large_data/shuttle.tst')[0:10]
print("Data loaded")

num_trials = 1
adaptive_results, hashing_results = compare_k_values(kernel_data, queries, num_trials)

# plot_k_comparison(adaptive_results, hashing_results)

for k, (results, avg_error, avg_time) in adaptive_results.items():
    print(f"\nAdaptive Shell Algorithm (k={k}):")
    print(f"Average execution time: {avg_time:.5f} seconds")
    print(f"Average overall error: {avg_error:.10f}")

print(f"\nHashing Based Algorithm:")
print(f"Average execution time: {hashing_results[2]:.5f} seconds")
print(f"Average overall error: {hashing_results[1]:.10f}")