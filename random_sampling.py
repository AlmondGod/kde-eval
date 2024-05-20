import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(x, sigma=1.0):
    return np.exp(-0.5 * (x)**2)

def student_kernel(x, sigma=1.0):
    return 1/((x)**2+1)

datasets = {
    'covtype': [54, 581012, 10000, 2.2499],
    'shuttle': [9, 43500, 43500, 0.621882],
}

kernel_data = np.loadtxt("data/our_data_askit.data", delimiter=',', skiprows=0)
test_data = np.loadtxt("data/our_data_askit_query.data", delimiter=',', skiprows=0)
print("Dimensions of kernel_data:", kernel_data.shape)
print("Dimensions of test_data:", test_data.shape)

# we use a subset of kernel_data for efficiency
# kernel_data = kernel_data[:10000]
kernel_data = np.random.choice(kernel_data, 10000, replace=False) #random sample to reduce kernel data
q_point = test_data[100]

def kernel_density_true(data, query):
    kde = 0
    for row in data:
        diff = query - row
        kde += gaussian_kernel(np.linalg.norm(diff)) / len(data)
    return kde

kde_true = kernel_density_true(kernel_data, q_point)
print("True KDE:", kde_true)

def kde_random_sampling(data, fun, query, num_samples):
    rand_kde = 0
    num_reps = 500
    for _ in range(num_reps):
        samples = np.random.choice(range(0, len(data)), num_samples, replace=False)
        kde = 0
        for i in samples:
            diff = query - data[i]
            kde += fun(np.linalg.norm(diff))
        rand_kde += kde / num_samples
    return rand_kde / num_reps

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

gloss = []
stloss = []
for T in T_values:
    gkde = kde_random_sampling(kernel_data, gaussian_kernel, q_point, T)
    skde = kde_random_sampling(kernel_data, student_t_kernel, q_point, T)
    gloss.append(np.abs((gkde - kde_true) / kde_true))
    stloss.append(np.abs((skde - kde_true) / kde_true))
    print(f"T={T}, Gaussian KDE={gkde}, gLoss={gloss[-1]}, Student-t KDE={skde}, stLoss={stloss[-1]} ")

output_file = "loss_output.txt"
with open(output_file, 'a') as f:
    f.write("exponential scale:")
    for l in gloss:
        f.write("gaussian loss:" + str(l) + '\n')  
    for l in stloss:
        f.write("student-t loss:" + str(l) + '\n')
print("Losses saved to:", output_file)

plt.plot(T_values, gloss, label='Gaussian Kernel Loss')
plt.plot(T_values, stloss, label='Student-t Kernel Loss')
plt.xlabel('Number of Samples (T)')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
