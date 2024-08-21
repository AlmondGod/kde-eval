import gzip
import csv
import numpy as np
import matplotlib.pyplot as plt

def student_kernel(x):
    return 1/((x)**2+1)

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
poker_data = np.genfromtxt('large_data/poker-hand-training-true.data', delimiter=',', dtype=int)

# with gzip.open('large_data/covtype.data.gz', 'rb') as f:
#     data = np.genfromtxt(f, delimiter=',', max_rows=1000000)

print("finished data loading")

datasets = {
    'higgs': (higgs_data, 3.41),
    'susy': (susy_data, 2.24),
    'skin': (skin_data, 0.24),
    'shuttle': (shuttle_data, 0.62),
    'sensorless': (sensorless_drive_data, 2.29),
    'home': (home_data, 0.53),
    'hep': (hep_data, 3.36),
    #'corel': (corel_data, 1.04),
    'poker': (poker_data, 1.37)
}

def compute_ratio_for_query(data, query, sigma):
    distances = np.linalg.norm(data - query, axis=1)
    k = student_kernel(distances / sigma)
    E_k = np.mean(k)
    E_k_sq = np.mean(k**2)
    return E_k_sq / (E_k**2)

n_queries = 1000

plt.figure(figsize=(20, 15))

for i, (name, (data, sigma)) in enumerate(datasets.items(), 1):
    print(name)
    query_indices = np.random.choice(data.shape[0], n_queries, replace=False)
    queries = data[query_indices]
    
    #we want this quantity to be high for a good amount of queries in the dataset
    ratios = [compute_ratio_for_query(data, query, sigma) for query in queries]

    ratios.sort()
    
    plt.subplot(3, 3, i)
    plt.plot(range(1, n_queries + 1), ratios, marker='o', linestyle='-', markersize=3)
    plt.title(f"{name} (σ={sigma})")
    plt.ylabel("E[k^2] / E[k]^2")
    plt.xlim(1, n_queries)
    plt.grid(True)

plt.tight_layout()
plt.show()

print("finish")