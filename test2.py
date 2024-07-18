import numpy as np
from scipy.spatial import KDTree
import time
from scipy.special import erf
import matplotlib.pyplot as plt
from demo.eLSH import ELSH

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

def student_kernel(x):
    return 1/((x)**2+1)
def adaptive_shell_algorithm(kernel_data, queries):
    print("Adaptive Shell algorithm start")
    dataset_size, dimensions = kernel_data.shape
    k = 1
    gaussian_vectors = np.random.normal(loc=0, scale=1, size=(k, dimensions))
    projected_data = np.dot(kernel_data, gaussian_vectors.T) / np.sqrt(k)
    projected_queries = np.dot(queries, gaussian_vectors.T) / np.sqrt(k)
    tree = KDTree(projected_data)

    num_spheres = 50
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

# Load data
kernel_data = np.loadtxt('large_data/shuttle.tst')
queries = np.loadtxt('large_data/shuttle.tst')[100:110]
print("Data loaded")

# Run both algorithms
adaptive_results, adaptive_overall_error, adaptive_time = adaptive_shell_algorithm(kernel_data, queries)
hashing_results, hashing_overall_error, hashing_time = hashing_based_algorithm(kernel_data, queries)

# Create plots
query_indices = range(1, len(queries) + 1)

# Plot 1: Errors
plt.figure(figsize=(12, 6))
plt.plot(query_indices, [r[2] for r in adaptive_results], 'bo-', label='Adaptive Shell')
plt.plot(query_indices, [r[2] for r in hashing_results], 'ro-', label='Hashing Based')
plt.xlabel('Query Index')
plt.ylabel('Error')
plt.title('Error Comparison: Adaptive Shell vs Hashing Based')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Computation Time
plt.figure(figsize=(12, 6))
plt.plot(query_indices, [r[3] for r in adaptive_results], 'bo-', label='Adaptive Shell')
plt.plot(query_indices, [r[3] for r in hashing_results], 'ro-', label='Hashing Based')
plt.xlabel('Query Index')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time Comparison: Adaptive Shell vs Hashing Based')
plt.legend()
plt.grid(True)
# plt.savefig('time_comparison.png')
# plt.close()
plt.show()

# Print results
print(f"Adaptive Shell Algorithm:")
print(f"Execution time: {adaptive_time:.2f} seconds")
print(f"Overall error: {adaptive_overall_error:.10f}")

print(f"\nHashing Based Algorithm:")
print(f"Execution time: {hashing_time:.2f} seconds")
print(f"Overall error: {hashing_overall_error:.10f}")

print("\nDetailed results:")
for i, (adaptive, hashing) in enumerate(zip(adaptive_results, hashing_results)):
    print(f"\nQuery {i+1}:")
    print(f"Adaptive Shell: Estimated: {adaptive[0]:.10f}, Actual: {adaptive[1]:.10f}, Error: {adaptive[2]:.10f}, Time: {adaptive[3]:.10f}")
    print(f"Hashing Based:  Estimated: {hashing[0]:.10f}, Actual: {hashing[1]:.10f}, Error: {hashing[2]:.10f}, Time: {hashing[3]:.10f}")