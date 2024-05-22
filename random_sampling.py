import numpy as np
import matplotlib.pyplot as plt
import time

def gaussian_kernel(x, sigma=1.0):
    return np.exp(-0.5 * (x)**2)

def student_kernel(x, sigma=1.0):
    return 1/((x)**2+1)

datasets = {
    'covtype': [54, 581012, 10000, 2.2499],
    'shuttle': [9, 43500, 43500, 0.621882],
}

kernel_data = np.loadtxt("data/covtype_normed.csv", delimiter=',', skiprows=0)[:, 1:]
test_data = np.loadtxt("data/shuttle_normed.csv", delimiter=',', skiprows=0)
print("Dimensions of kernel_data:", kernel_data.shape)
print("Dimensions of test_data:", test_data.shape)

# we use a subset of kernel_data for efficiency
# kernel_data = kernel_data[:10000]
kernel_data = np.random.choice(range(1, len(kernel_data)), 10000, replace=False) #random sample to reduce kernel data
print(len(kernel_data))
q_point = kernel_data[0]

def kernel_density_true(data, query):
    kde = 0
    for row in data:
        diff = query - row
        kde += gaussian_kernel(np.linalg.norm(diff)) / len(data)
    return kde

kde_true = kernel_density_true(kernel_data, q_point)
print("True KDE:", kde_true)

def kde_random_sampling(data, fun, query, num_samples):
    kde_estimates = []
    num_reps = 500
    for _ in range(num_reps):
        samples = np.random.choice(range(0, len(data)), num_samples, replace=False)
        kde = 0
        for i in samples:
            diff = query - data[i]
            kde += fun(np.linalg.norm(diff))
        kde_estimates.append(kde / num_samples)
    mean_kde = np.mean(kde_estimates)
    var_kde = np.var(kde_estimates)
    return mean_kde, var_kde

def generate_sampling_points(max_value, base=2):
    T_values = []
    T = 1
    while T <= max_value:
        T_values.append(T)
        T *= base
    return np.array(T_values)

max_samples = 10000
T_values = generate_sampling_points(max_samples)
print("Sampling points (T values):", T_values)

grpe = []
strpe = []
gaussian_times = []
student_times = []
gaussian_variances = []
student_variances = []

for T in T_values:
    start_time = time.time()
    gkde, gvar = kde_random_sampling(kernel_data, gaussian_kernel, q_point, T)
    gaussian_times.append(time.time() - start_time)
    gaussian_variances.append(gvar)

    start_time = time.time()
    skde, svar = kde_random_sampling(kernel_data, student_kernel, q_point, T)
    student_times.append(time.time() - start_time)
    student_variances.append(svar)

    grpe.append(np.abs((gkde - kde_true) / kde_true))
    strpe.append(np.abs((skde - kde_true) / kde_true))
    print(f"T={T}, Gaussian KDE={gkde}, gRPE={grpe[-1]}, Student-t KDE={skde}, stRPE={strpe[-1]} ")

output_file = "RPE_output.txt"
with open(output_file, 'a') as f:
    for i, t in enumerate(T_values):
        f.write(f"At t = {t}, gaussian RPE={grpe[i]} and student RPE={strpe[i]} \n" + 
                f"with gaussian time={gaussian_times[i]} and student time={student_times[i]}\n")  
print("RPEs saved to:", output_file)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].plot(T_values, grpe, label='Gaussian Kernel Loss')
axs[0].plot(T_values, strpe, label='Student-t Kernel Loss')
axs[0].set_xlabel('Number of Samples (T)')
axs[0].set_ylabel('RPE')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].set_title('Kernel RPE')

axs[1].plot(T_values, gaussian_times, label='Gaussian Kernel Time')
axs[1].plot(T_values, student_times, label='Student-t Kernel Time')
axs[1].set_xlabel('Number of Samples (T)')
axs[1].set_ylabel('Time (seconds)')
axs[1].set_xscale('log')
axs[1].legend()
axs[1].set_title('Kernel Computation Time')

axs[2].plot(T_values, gaussian_variances, label='Gaussian Kernel Variance')
axs[2].plot(T_values, student_variances, label='Student-t Kernel Variance')
axs[2].set_xlabel('Number of Samples (T)')
axs[2].set_ylabel('Variance')
axs[2].set_xscale('log')
axs[2].legend()
axs[2].set_title('Kernel Variance')

plt.tight_layout()
plt.show()