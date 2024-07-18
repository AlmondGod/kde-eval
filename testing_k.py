import numpy as np
from scipy.spatial import KDTree
import time
import matplotlib.pyplot as plt
from demo.eLSH import ELSH

def student_kernel(x):
    return 1/((x)**2+1)

def adaptive_shell_algorithm(kernel_data, queries, k=1, num_spheres=50):
    print(f"Adaptive Shell algorithm start with k={k}")
    dataset_size, dimensions = kernel_data.shape
    gaussian_vectors = np.random.normal(loc=0, scale=1, size=(k, dimensions))
    projected_data = np.dot(kernel_data, gaussian_vectors.T) / np.sqrt(k)
    projected_queries = np.dot(queries, gaussian_vectors.T) / np.sqrt(k)
    tree = KDTree(projected_data)

    max_distance = np.max(projected_data) - np.min(projected_data)
    radii = np.linspace(0, max_distance, num_spheres + 1)[1:]

    def estimate_kernel_squared(query):
        kernel_sq_estimate = 0
        points_counted = 0
        for i, radius in enumerate(radii):
            if i == 0:
                count = tree.query_ball_point(query.reshape(1, -1), radius, return_length=True)
            else:
                count = tree.query_ball_point(query.reshape(1, -1), radius, return_length=True) - \
                        tree.query_ball_point(query.reshape(1, -1), radii[i-1], return_length=True)
            
            if count > 0:
                sample_distance = (radius + (radii[i-1] if i > 0 else 0)) / 2
                kernel_sq_estimate += count * student_kernel(sample_distance)**2
            points_counted += count
        
        return kernel_sq_estimate / points_counted if points_counted > 0 else 0

    epsilon = 0.5
    results = []
    query_times = []

    for q in queries:
        query_start_time = time.time()
        variance = estimate_kernel_squared(projected_queries[np.where((queries == q).all(axis=1))[0][0]])
        
        j = 1
        r = np.random.randint(0, dataset_size-1)
        kernel_sumStudent = student_kernel(np.linalg.norm(kernel_data[r] - q))
        averageStudent = kernel_sumStudent/j
        t = variance / ((averageStudent)**2 * (epsilon)**2)

        while j < t:
            r = np.random.randint(0, dataset_size-1)
            kernel_sumStudent += student_kernel(np.linalg.norm(kernel_data[r] - q))
            j += 1
            averageStudent = kernel_sumStudent/j
            t = variance / (epsilon**2 * averageStudent**2)
                
        averageStudent = kernel_sumStudent/j
        actual_kernel_sq = np.mean([student_kernel(np.linalg.norm(x - q))**2 for x in kernel_data])
        percent_error = np.abs(averageStudent - actual_kernel_sq) / actual_kernel_sq
        query_time = time.time() - query_start_time
        results.append((averageStudent, actual_kernel_sq, percent_error, query_time))

    overall_error = np.mean([r[2] for r in results])
    execution_time = sum([r[3] for r in results])

    return results, overall_error, execution_time

def hashing_based_algorithm(kernel_data, queries):
    print("Hashing Based algorithm start")
    kde = GHBE(kernel_data.T, 
           tau=1e-2, eps=0.2, gamma=0.6,
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
    k_values = [1, 2, 3]
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
    
    # Prepare data for plotting
    errors = [result[1] for result in adaptive_results.values()]
    times = [result[2] for result in adaptive_results.values()]
    
    hashing_error = hashing_results[1]
    hashing_time = hashing_results[2]
    
    # Plot 1: Errors
    plt.figure(figsize=(12, 6))
    plt.bar(k_values, errors, label='Adaptive Shell')
    plt.axhline(y=hashing_error, color='r', linestyle='--', label='Hashing Based')
    plt.xlabel('k value')
    plt.ylabel('Average Overall Error')
    plt.title('Error Comparison: Adaptive Shell (varying k) vs Hashing Based')
    plt.legend()
    plt.show()
    
    # Plot 2: Execution Times
    plt.figure(figsize=(12, 6))
    plt.bar(k_values, times, label='Adaptive Shell')
    plt.axhline(y=hashing_time, color='r', linestyle='--', label='Hashing Based')
    plt.xlabel('k value')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Execution Time Comparison: Adaptive Shell (varying k) vs Hashing Based')
    plt.legend()
    plt.show()
    
    # Plot 3: Error vs Time Scatter Plot
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

# Load data
kernel_data = np.loadtxt('large_data/shuttle.tst')
queries = np.loadtxt('large_data/shuttle.tst')
print("Data loaded")

# Run comparison with multiple trials
num_trials = 5  # You can adjust this number
adaptive_results, hashing_results = compare_k_values(kernel_data, queries, num_trials)

# Plot results
plot_k_comparison(adaptive_results, hashing_results)

# Print detailed results
for k, (results, avg_error, avg_time) in adaptive_results.items():
    print(f"\nAdaptive Shell Algorithm (k={k}):")
    print(f"Average execution time: {avg_time:.2f} seconds")
    print(f"Average overall error: {avg_error:.10f}")

print(f"\nHashing Based Algorithm:")
print(f"Average execution time: {hashing_results[2]:.2f} seconds")
print(f"Average overall error: {hashing_results[1]:.10f}")