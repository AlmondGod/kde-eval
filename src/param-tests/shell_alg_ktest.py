import numpy as np

def student_kernel(x):
    return 1/((x)**2+1)

#import dataset and queryset
kernel_data = np.loadtxt("data/testing_askit.data", delimiter=',', skiprows=0)[:, 1:11]
queries = np.loadtxt("data/testing_askit_query.data", delimiter=',', skiprows=0)[:10, 1:11]
print("data loaded")
dataset_size = kernel_data.shape[0]
dimensions = kernel_data.shape[1]

#reduce dimensionality of data to k-dimensions with dot products with K d-length gaussian vectors and each datapoint
k = 1
gaussian_vectors = np.random.normal(loc=0, scale=1, size=(k, dimensions))

projected_data = np.dot(kernel_data, gaussian_vectors.T) / np.sqrt(k)
projected_queries = np.dot(queries, gaussian_vectors.T) / np.sqrt(k)

#kernel squared of query and each datapoint for both original and projected
oned_kernelsq = 0
kernel_squared = 0
overall_error = 0
highest_error = 0
best_error = 100
for q in range(len(queries)):
    for i in range(dataset_size):
        oned_kernelsq += student_kernel(np.linalg.norm(projected_data[i] - projected_queries[q]))**2
        kernel_squared += student_kernel(np.linalg.norm(kernel_data[i] - queries[q]))**2
        
    oned_kernelsq = oned_kernelsq / dataset_size
    kernel_squared = kernel_squared / dataset_size
    # print(oned_kernelsq, kernel_squared)
    percent_error = np.abs(oned_kernelsq - kernel_squared) / kernel_squared
    overall_error += percent_error
    if percent_error > highest_error:
        highest_error = percent_error
    if percent_error < best_error:
        best_error = percent_error
    print(oned_kernelsq, kernel_squared, percent_error)

overall_error = overall_error / len(queries)
with open("shell_alg_ktest_results.txt", "a") as f:
    f.write(f"{dimensions} to {k}, averaged error: {overall_error}, low: {best_error}, hi: {highest_error}\n")
print(f"Overall error: {overall_error}")

