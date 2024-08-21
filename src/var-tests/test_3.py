import numpy as np
from scipy.spatial import KDTree
import time
import matplotlib.pyplot as plt
from hbe.eLSH import ELSH
import gzip

def student_kernel(x):
    return 1/((x)**2+1)

def adaptive_shell_algorithm(kernel_data, queries, k=1, num_spheres=50):
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
    query_times = []
    sample_counts = []

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
                
        query_time = time.time() - query_start_time
        query_times.append(query_time)
        sample_counts.append(j)

    return query_times, sample_counts

def hashing_based_algorithm(kernel_data, queries):
    kde = GHBE(kernel_data.T, 
           tau=1e-2, eps=0.2, gamma=0.6,
           max_levels=8, max_tables_per_level=500,
           k_factor=2, w_factor=1.0)

    query_times = []
    for q in queries:
        query_start_time = time.time()
        kde.AMR(q)
        query_time = time.time() - query_start_time
        query_times.append(query_time)

    return query_times

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

def plot_results(adaptive_times, hashing_times, sample_counts):
    # Plot 1: Execution times comparison
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(adaptive_times)), adaptive_times, label='Adaptive Shell')
    plt.plot(range(len(hashing_times)), hashing_times, label='Hashing')
    plt.xlabel('Query Index')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison: Adaptive Shell vs Hashing')
    plt.legend()
    plt.show()

    # Plot 2: Number of samples for Adaptive Shell
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sample_counts)), sample_counts)
    plt.xlabel('Query Index')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples Taken by Adaptive Shell Algorithm for Each Query')
    plt.show()


with gzip.open('large_data/HIGGS.csv.gz', 'rb') as f:
    higgs_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
with gzip.open('large_data/SUSY.csv.gz', 'rb') as f:
    susy_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
with gzip.open('large_data/all_train.csv.gz', 'rb') as f:
    hep_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
# with gzip.open('large_data/ColorHistogram.asc.gz', 'rb') as f:
#     corel_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
print("halfway")
shuttle_data = np.loadtxt('large_data/shuttle.tst')
sensorless_drive_data = np.loadtxt('large_data/sensorless_drive_diagnosis.txt')
home_data = np.genfromtxt('large_data/HT_Sensor_dataset.dat', skip_header=1, delimiter=None)
skin_data = np.loadtxt('large_data/Skin_NonSkin.txt')

datasets = {
    'higgs': (higgs_data, 3.41),
    'susy': (susy_data, 2.24),
    'skin': (skin_data, 0.24),
    'shuttle': (shuttle_data, 0.62),
    'sensorless': (sensorless_drive_data, 2.29),
    'home': (home_data, 0.53),
    'hep': (hep_data, 3.36),
    # 'corel': (corel_data, 1.04)
    # 'covtype': (data, 2.25)
}
# Load data
kernel_data = np.loadtxt('large_data/shuttle.tst')
queries = np.loadtxt('large_data/shuttle.tst')
print("Data loaded")

# Run algorithms
adaptive_times, sample_counts = adaptive_shell_algorithm(kernel_data, queries)
print("running hashing")
# hashing_times = hashing_based_algorithm(kernel_data, queries)

# Plot results
# plot_results(adaptive_times, hashing_times, sample_counts)
plot_results(adaptive_times, [], sample_counts)

# Print summary statistics
print(f"\nAdaptive Shell Algorithm:")
print(f"Average execution time: {np.mean(adaptive_times):.2f} seconds")
print(f"Average number of samples: {np.mean(sample_counts):.2f}")

print(f"\nHashing Based Algorithm:")
# print(f"Average execution time: {np.mean(hashing_times):.2f} seconds")