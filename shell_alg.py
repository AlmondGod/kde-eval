import numpy as np
from scipy.spatial import KDTree

def student_kernel(x):
    return 1/((x)**2+1)

kernel_data = np.loadtxt("data/testing_askit.data", delimiter=',', skiprows=0)[:, 1:]
queries = np.loadtxt("data/testing_askit_query.data", delimiter=',', skiprows=0)[900:910, 1:]
print("Data loaded")
dataset_size = kernel_data.shape[0]
dimensions = kernel_data.shape[1]

#reduce dimensionality of data to k-dimensions with dot products with K d-length gaussian vectors and each datapoint
k = 5
gaussian_vectors = np.random.normal(loc=0, scale=1, size=(k, dimensions))

projected_data = np.dot(kernel_data, gaussian_vectors.T) / np.sqrt(k)
projected_queries = np.dot(queries, gaussian_vectors.T) / np.sqrt(k)

tree = KDTree(projected_data) #just a binary tree that splits data along median of certain dimension repeating every k dims

num_spheres = 50 #how many intervals along the integral to take

#defining it this way for testing purposes but probs need to find a better way to get num intervals 
max_distance = np.max(projected_data) - np.min(projected_data)
sphere_radii = np.linspace(0, max_distance, num_spheres + 1)[1:]
print(sphere_radii)

def estimate_kernel_squared(tree, query, radii):
    kernel_sq_estimate = 0
    points_counted = 0
    for i in range(len(radii)): #basically getting number of points in the ring shell then mulitiplying by mid distance (tried first distace and the overestimation was way too large)
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
    
    return kernel_sq_estimate / points_counted

overall_error = 0

for q in range(len(queries)):
    estimated_kernel_sq = estimate_kernel_squared(tree, projected_queries[q], sphere_radii)
    actual_kernel_sq = np.mean([student_kernel(np.linalg.norm(kernel_data[i] - queries[q]))**2 for i in range(dataset_size)])
    percent_error = np.abs(estimated_kernel_sq - actual_kernel_sq) / actual_kernel_sq
    overall_error += percent_error
    
    print(f"Estimated: {estimated_kernel_sq}, Actual: {actual_kernel_sq}, Error: {percent_error}")
    with open("shell_alg_results.txt", "a") as f:
        if estimated_kernel_sq < actual_kernel_sq:
            f.write("Underestimated, ")
        else:
            f.write("Overestimated, ")
        f.write(f"Estimated: {estimated_kernel_sq}, Actual: {actual_kernel_sq}, Error: {percent_error}\n")

overall_error /= len(queries)

with open("shell_alg_results.txt", "a") as f:
    f.write(f"{dimensions} to {k}, {num_spheres} spheres, averaged error: {overall_error}\n\n")
print(f"Overall error: {overall_error}")