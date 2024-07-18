import numpy as np
from scipy.spatial import KDTree

def student_kernel(x):
    return 1/((x)**2+1)

kernel_data = np.loadtxt('large_data/shuttle.tst')
queries = np.loadtxt('large_data/shuttle.tst')[100:110]
print("Data loaded")
dataset_size = kernel_data.shape[0]
dimensions = kernel_data.shape[1]

#reduce dimensionality of data to k-dimensions with dot products with K d-length gaussian vectors and each datapoint
k = 2
gaussian_vectors = np.random.normal(loc=0, scale=1, size=(k, dimensions))

projected_data = np.dot(kernel_data, gaussian_vectors.T) / np.sqrt(k)
projected_queries = np.dot(queries, gaussian_vectors.T) / np.sqrt(k)

tree = KDTree(projected_data)

num_spheres = 50 #how many intervals along the integral to take

#defining it this way for testing purposes but probs need to find a better way to get num intervals 
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

#kernel squared of query and each datapoint for both original and projected
overall_error = 0
overall_error2 = 0
# highest_error = 0
# best_error = 100000
epsilon = 0.1
import random
for q in range(len(queries)):
    variance = estimate_kernel_squared(projected_queries[q])
    
    j = 1
    r = random.randint(0, dataset_size-1)
    diff = kernel_data[r] - queries[q]
    kernel_sumStudent = student_kernel(np.linalg.norm(diff))
    averageStudent = kernel_sumStudent/j
    t = variance / ((averageStudent)**2 * (epsilon)**2)
    print(f"t: {t}")

    while (j < t):
        r = random.randint(0, 581011)
        diff = kernel_data[r] - queries[q]
        kernel_sumStudent += student_kernel(np.linalg.norm(diff))
                
        j += 1
        averageStudent = kernel_sumStudent/j
        condition = variance / (epsilon**2  * averageStudent)
            
    print(j)
    averageStudent = kernel_sumStudent/j
    
    actual_kernel_sq = np.mean([student_kernel(np.linalg.norm(kernel_data[i] - queries[q]))**2 for i in range(dataset_size)])
    
    percent_error = np.abs(averageStudent - actual_kernel_sq) / actual_kernel_sq
    overall_error += percent_error

    print(f"variance: {variance}, T: {j}, Estimated: {averageStudent}, Actual: {actual_kernel_sq}, Error: {percent_error}")

overall_error /= len(queries)

with open("shell_adaptive_results.txt", "a") as f:
    f.write(f"T: {t}, Estimated: {averageStudent}, Actual: {actual_kernel_sq}, Error: {percent_error}")