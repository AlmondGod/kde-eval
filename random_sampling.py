import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as erf
from functools import partial as partial
import numpy as np

def gaussian_kernel(x, sigma=1.0):
    return np.exp(-0.5 * (x / sigma)**2)

datasets = {
	'covtype': [54, 581012, 10000, 2.2499],
	'shuttle': [9, 43500, 43500, 0.621882],

}

kernel_data = np.loadtxt("data/our_data_askit.data", delimiter=',', skiprows=0)
test_data = np.loadtxt("data/our_data_askit_query.data", delimiter=',', skiprows=0)
print("Dimensions of kernel_data:", kernel_data.shape)
print("Dimensions of test_data:", test_data.shape)

kernel_data = kernel_data[:10000] #for time efficiency's sake, can uncomment
q_point = test_data[0]

def kernel_density_true(data, query):
    kde = 0
    for row in data:
        diff = query - row
        kde += gaussian_kernel(np.linalg.norm(diff)) / len(kernel_data)
    return kde

kde_true = kernel_density_true(kernel_data, q_point)
print(kde_true)

def kde_random_sampling(data, query, num_samples):
    rand_kde = 0
    num_reps = 100
    for _ in range(num_reps):
        samples = np.random.choice(range(0, len(kernel_data)), num_samples)
        kde = 0
        for i in samples:
            diff = query - data[i]
            kde += gaussian_kernel(np.linalg.norm(diff)) / (num_reps * num_samples)
        rand_kde += kde
    return rand_kde

loss = []

exponents = np.arange(0, 19)
exp_array = 2 ** exponents

lin_array = range(0, len(kernel_data), int(len(kernel_data) / 40))

for i in exp_array:
    kde = kde_random_sampling(kernel_data, q_point, int(i))
    print(str(i) + " has: " + str(kde))
    loss.append(np.abs(kde - kde_true))

output_file = "loss_output.txt"
with open(output_file, 'a') as f:
    for l in loss:
        f.write(str(l) + '\n')  
print("Losses saved to:", output_file)

plt.plot(loss)
plt.show()