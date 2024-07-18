import gzip
import csv
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(x, sigma=1.0):
    return np.exp(-(x)**2 / (2 * (sigma**2)))

print("loading data")
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

def compute_kernel_squared(data, query, sigma):
    distances = np.linalg.norm(data - query, axis=1)
    k = gaussian_kernel(distances, sigma)
    return np.mean(k**2)

def compute_kernel_squared_1d(data, query, sigma):
    distances = np.abs(data - query)
    k = gaussian_kernel(distances, sigma)
    return np.mean(k**2)

def reduce_dim(data):
    k = 1
    gaussian_vectors = np.random.normal(loc=0, scale=1, size=(k, data.shape[1]))
    projected_data = np.dot(data, gaussian_vectors.T) / np.sqrt(k)
    return projected_data

n_queries = 100

plt.figure(figsize=(20, 15))

for i, (name, (data, sigma)) in enumerate(datasets.items(), 1):
    print(name)

    query_indices = np.random.choice(data.shape[0], n_queries, replace=False)
    queries = data[query_indices]
    data_1d = np.zeros((data.shape[0], 1))

    for j in range(10):
        data_1d += reduce_dim(data)
    
    data_1d /= 10

    oned_queries = data_1d[query_indices]

    regular_k_squared = [compute_kernel_squared(data, query, sigma) for query in queries]
    projected_k_squared = [compute_kernel_squared_1d(data_1d, query, sigma) for query in oned_queries]
    percent_errors = [p / r for r, p in zip(regular_k_squared, projected_k_squared)]

    count_less_than_one = sum(1 for pe in percent_errors if pe < 1)
    print(f"Number of items where percent error < 1: {count_less_than_one}")

    count_greater_than_one = sum(1 for pe in percent_errors if pe > 1)
    print(f"Number of items where percent error > 1: {count_greater_than_one}")
    # print(f"regular kernel sq:  {regular_k_squared}")
    # print(f"estimates: {projected_k_squared}")

    percent_errors.sort()
    print(f"percent errors lowest ten: {[x.item() for x in percent_errors[:10]]}")
    plt.subplot(3, 3, i)
    plt.plot(range(1, n_queries + 1), percent_errors, marker='o', linestyle='-', markersize=3)
    plt.title(f"{name} (σ={sigma})")
    plt.xlabel("Query Index")
    plt.ylabel("Ratio original / projected")
    plt.xlim(1, n_queries)
    plt.grid(True)

plt.tight_layout()
plt.show()

print("finished computing ratios")