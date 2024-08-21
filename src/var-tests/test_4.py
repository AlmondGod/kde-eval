import numpy as np
from scipy.spatial import KDTree
import time
import matplotlib.pyplot as plt
from hbe.eLSH import ELSH
import gzip

def gaussian_kernel(x, sigma=1.0):
    # print(-(x)**2 / ((2 * (sigma**2))))
    # print(f"gauss kernel: {np.exp(-(x)**2 / ((2 * (sigma**2))))}")
    return np.exp(-(x)**2 / ((2 * (sigma**2))))

def adaptive_shell_algorithm(kernel_data, queries, sigma, k=1, num_spheres=50):
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
                kernel_sq_estimate += count * gaussian_kernel(sample_distance, sigma)**2
            points_counted += count
        
        return kernel_sq_estimate / points_counted if points_counted > 0 else 0

    epsilon = 0.5
    query_times = []
    sample_counts = []

    for i, q in enumerate(projected_queries):
        query_start_time = time.time()
        variance = estimate_kernel_squared(q)
        
        j = 1
        r = np.random.randint(0, dataset_size-1)
        kernel_sum = gaussian_kernel(np.linalg.norm(kernel_data[r] - queries[i]), sigma)
        # print(f"kernel_sum: {kernel_sum}, projected_data[r]: {kernel_data[r]}, q: {queries[i]}")
        average = kernel_sum / j
        t = variance / ((average) * (epsilon)**2)
        # print(np.linalg.norm(kernel_data[r] - queries[i]))
        # print(sigma)
        print(f"t: {t}, variance: {variance}, average: {average}")

        while j < t:
            r = np.random.randint(0, dataset_size-1)
            kernel_sum += gaussian_kernel(np.linalg.norm(kernel_data[r] - queries[i]), sigma)
            j += 1
            average = kernel_sum / j
            t = variance / (epsilon**2 * average)
                
        query_time = time.time() - query_start_time
        query_times.append(query_time)
        sample_counts.append(j)

    return query_times, sample_counts

def plot_results(dataset_name, sigma, adaptive_times, sample_counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{dataset_name} (σ={sigma})')

    # Plot 1: Execution times
    ax1.plot(range(len(adaptive_times)), adaptive_times)
    ax1.set_xlabel('Query Index')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time for Adaptive Shell')

    # Plot 2: Number of samples for Adaptive Shell
    ax2.plot(range(len(sample_counts)), sample_counts)
    ax2.set_xlabel('Query Index')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Number of Samples Taken by Adaptive Shell Algorithm')

    plt.tight_layout()
    return fig

print("Loading data...")
with gzip.open('large_data/HIGGS.csv.gz', 'rb') as f:
    higgs_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
with gzip.open('large_data/SUSY.csv.gz', 'rb') as f:
    susy_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
with gzip.open('large_data/all_train.csv.gz', 'rb') as f:
    hep_data = np.genfromtxt(f, delimiter=',', max_rows=1000000)
print("Halfway...")
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
}

print("Data loaded")

# Create a grid for plots
n_datasets = len(datasets)
n_cols = 3
n_rows = (n_datasets + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
axes = axes.flatten()

for i, (dataset_name, (data, sigma)) in enumerate(datasets.items()):
    print(f"Processing {dataset_name}...")
    n_queries = 100
    query_indices = np.random.choice(data.shape[0], n_queries, replace=False)
    queries = data[query_indices]

    adaptive_times, sample_counts = adaptive_shell_algorithm(data, queries, sigma)

    # Plot results for this dataset
    plot = plot_results(dataset_name, sigma, adaptive_times, sample_counts)
    
    # Add the plot to the grid
    axes[i].imshow(plot.canvas.renderer._renderer)
    axes[i].axis('off')
    plt.close(plot)

    print(f"\n{dataset_name} (σ={sigma}):")
    print(f"Average execution time: {np.mean(adaptive_times):.2f} seconds")
    print(f"Average number of samples: {np.mean(sample_counts):.2f}")

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

print("Finished processing all datasets")