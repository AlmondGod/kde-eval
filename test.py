import numpy as np
from scipy.spatial import KDTree
import time
from scipy.special import erf
from demo.eLSH import ELSH

class GHBE:
    def __init__(self, X, tau=1e-3, eps=0.1, gamma=0.5, max_levels=10, max_tables_per_level=1000, k_factor=3, w_factor=np.sqrt(2.0/np.pi)):
        self.eps = eps
        self.R = np.sqrt(np.log(1 / tau))  # effective diameter of the set
        self.gamma = gamma  # rate that we decrease our estimate
        
        # Limit the number of levels
        self.I = min(int(np.ceil(np.log2(1 / tau))), max_levels)
        
        self.mui = np.array([(1 - self.gamma) ** i for i in range(self.I)])
        self.ti = np.sqrt(np.log(1/self.mui)) / 2.0  # nominal scale for level i
        
        # Adjust ki calculation
        self.ki = [int(k_factor * np.ceil(self.R * self.ti[j])) for j in range(self.I)]
        
        # Adjust wi calculation
        self.wi = self.ki / np.maximum(self.ti, 1) * w_factor
        
        # Simplified RelVar function
        self.RelVar = lambda mu: 1.0 / np.power(mu, 0.75)
        
        # Limit the number of hash tables per level
        self.Mi = [min(int(np.ceil(eps**-2 * self.RelVar(self.mui[j]))), max_tables_per_level) for j in range(self.I)]
        
        print(f"Building {sum(self.Mi)} hash functions across {self.I} levels")
        self.HTA = [ELSH(self.Mi[j], X, self.wi[j], self.ki[j],
                         lambda x, y: np.exp(-np.linalg.norm(x-y)**2))
                    for j in range(self.I)]
    
    def AMR(self, q):
            """
            Adaptive Mean Relaxation to figure out a constant factor approximation
            to the density KDE(q).
            """

            # Mean Relaxation
            for i in range(self.I):
                Z = 0.0
                for j in range(self.Mi[i]):
                    Z = Z + self.HTA[i].evalquery(q)
                print('Iteration {:d}, {:.10f} ? {:.10f}'.format(i,Z / (self.Mi[i]+0.0), self.mui[i] ))
                if Z / (self.Mi[i]+0.0)  >= self.mui[i]:
                    print('Success {:d}'.format(i))
                    return Z / (self.Mi[i])
            return Z / (self.Mi[i])
    
def student_kernel(x):
    return 1/((x)**2+1)

# Load data
# with gzip.open('large_data/HIGGS.csv.gz', 'rb') as f:
#     higgs_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
# with gzip.open('large_data/SUSY.csv.gz', 'rb') as f:
#     susy_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
# with gzip.open('large_data/all_train.csv.gz', 'rb') as f:
#     hep_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
# print("halfway")
# shuttle_data = np.loadtxt('large_data/shuttle.tst')
# sensorless_drive_data = np.loadtxt('large_data/sensorless_drive_diagnosis.txt')
# home_data = np.genfromtxt('large_data/HT_Sensor_dataset.dat', skip_header=1, delimiter=None)
skin_data = np.loadtxt('large_data/Skin_NonSkin.txt')
# kernel_data = np.loadtxt('large_data/shuttle.tst')
# queries = np.loadtxt('large_data/shuttle.tst')[100:110]
kernel_data = skin_data
queries = skin_data[1000:1010]
print("Data loaded")
datasets = {
    # 'higgs': (higgs_data, 3.41),
    # 'susy': (susy_data, 2.24),
    'skin': (skin_data, 0.24),
    # 'shuttle': (shuttle_data, 0.62),
    # 'sensorless': (sensorless_drive_data, 2.29),
    # 'home': (home_data, 0.53),
    # 'hep': (hep_data, 3.36),
}

sigma = 0.24

def adaptive_shell_algorithm(kernel_data, queries):
    print("adaptive start")
    dataset_size = kernel_data.shape[0]
    dimensions = kernel_data.shape[1]

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
        for i in range(len(radii)):
            if i == 0:
                count = tree.query_ball_point(query, radii[i], return_length=True)
            else:
                inner_count = tree.query_ball_point(query, radii[i-1], return_length=True)
                outer_count = tree.query_ball_point(query, radii[i], return_length=True)
                count = outer_count - inner_count
            
            if count > 0:
                sample_distance = (radii[i] + (radii[i-1] if i > 0 else 0)) / 2
                kernel_sq_estimate += count * student_kernel(sample_distance)**2
            points_counted += count
        
        return kernel_sq_estimate / points_counted if points_counted > 0 else 0

    overall_error = 0
    epsilon = 0.5
    start_time = time.time()
    results = []

    for q in range(len(queries)):
        variance = estimate_kernel_squared(projected_queries[q])
        
        j = 1
        r = np.random.randint(0, dataset_size-1)
        diff = kernel_data[r] - queries[q]
        kernel_sumStudent = student_kernel(np.linalg.norm(diff))
        averageStudent = kernel_sumStudent/j
        t = variance / ((averageStudent) * (epsilon)**2)

        while (j < t):
            r = np.random.randint(0, dataset_size-1)
            diff = kernel_data[r] - queries[q]
            kernel_sumStudent += student_kernel(np.linalg.norm(diff))
                    
            j += 1
            averageStudent = kernel_sumStudent/j
            t = variance / (epsilon**2 * averageStudent)
                
        averageStudent = kernel_sumStudent/j
        
        actual_kernel_sq = np.mean([student_kernel(np.linalg.norm(kernel_data[i] - queries[q]))**2 for i in range(dataset_size)])
        
        percent_error = np.abs(averageStudent - actual_kernel_sq) / actual_kernel_sq
        overall_error += percent_error
        print("variance: ", variance)
        print("num guesses: ", j)   

        results.append((averageStudent, actual_kernel_sq, percent_error))

    overall_error /= len(queries)
    execution_time = time.time() - start_time

    return results, overall_error, execution_time

def hashing_based_algorithm(kernel_data, queries):
    print("hbe start")
    tau = 1e-3
    eps = 0.5
    
    # kde = GHBE(kernel_data.T, tau, eps)
    kde = GHBE(kernel_data.T, 
           tau=1e-2,  # Increased from 1e-3, reduces levels
           eps=0.2,   # Increased from 0.1, reduces hash tables
           gamma=0.6, # Increased from 0.5, reduces levels
           max_levels=8,  # Explicitly limit levels
           max_tables_per_level=500,  # Limit tables per level
           k_factor=2,  # Reduced from 3, reduces hash functions per table
           w_factor=1.0  # Adjusted from sqrt(2/pi), affects bucket width
          )

    start_time = time.time()
    results = []

    for q in queries:
        est = kde.AMR(q)
        actual = np.mean([np.exp(-np.linalg.norm(q - x)**2) for x in kernel_data])
        percent_error = np.abs(est - actual) / actual
        results.append((est, actual, percent_error))

    overall_error = np.mean([r[2] for r in results])
    execution_time = time.time() - start_time

    return results, overall_error, execution_time


adaptive_results, adaptive_overall_error, adaptive_time = adaptive_shell_algorithm(kernel_data, queries)
hashing_results, hashing_overall_error, hashing_time = hashing_based_algorithm(kernel_data, queries)

print(f"Adaptive Shell Algorithm:")
print(f"Execution time: {adaptive_time:.2f} seconds")
print(f"Overall error: {adaptive_overall_error:.10f}")

print(f"\nHashing Based Algorithm:")
print(f"Execution time: {hashing_time:.2f} seconds")
print(f"Overall error: {hashing_overall_error:.10f}")

print("\nDetailed results:")
for i, (adaptive, hashing) in enumerate(zip(adaptive_results, hashing_results)):
    print(f"\nQuery {i+1}:")
    print(f"Adaptive Shell: Estimated: {adaptive[0]:.10f}, Actual: {adaptive[1]:.10f}, Error: {adaptive[2]:.10f}")
    print(f"Hashing Based:  Estimated: {hashing[0]:.10f}, Actual: {hashing[1]:.10f}, Error: {hashing[2]:.10f}")