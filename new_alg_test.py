from demo.eLSH import GHBE
from matthew_random_sample import a, b
import numpy as np
import random


def gaussian_kernel(x, sigma=1.0):
    return np.exp(-0.5 * (x)**2)


def true_kernel_density(data, query, kernel_fun):
    kernel_sum = 0
    for i in range(len(data)):
        kernel_sum += kernel_fun(data[i], query)
    return kernel_sum / len(data)

def new_alg (delta, epsilon, query, data, var_kernel, kernel_fun):
    tau = 0.1 / np.sqrt(len(data))
    print(f"tau: {tau}")
    points = np.array(data).transpose()
    print(f"points shape: {points.shape[1]}")
    ghbe = GHBE(points, tau, eps=1, kernel_fun=var_kernel)
    var_est = ghbe.AMR(query)
    print(f"variance estimate: {var_est}")
    true_var = true_kernel_density(data, query, var_kernel)
    print(f"true variance: {true_var}")

    j = 0
    averageGauss = 1    #averageStudent = 1
    condition = 1 / (epsilon**2 * delta * averageGauss)
    kernel_sumGauss = 0

    while j < len(data):
        r = random.randint(0, len(data) - 1)
        # diff = data[r] - query
        kernel_sumGauss += kernel_fun(data[r], query)
        
        j += 1
        averageGauss = kernel_sumGauss/j
        
        condition = 2 / (epsilon**2  * averageGauss)

        T = var_est / (epsilon**2 * averageGauss)
        # print(f"j: {j}, current estimate: {averageGauss}, current T: {T}")
        if (j > T):
            break
        
    averageGauss = kernel_sumGauss/30000

    return averageGauss

var_kernel = lambda x,y: np.exp(-0.125 * np.linalg.norm(x-y)**2)
kernel_fun = lambda x,y: np.exp(-0.5 * np.linalg.norm(x-y)**2)

query = b[0]
data = a[:1000]
delta = 0.1
epsilon = 0.1
new_alg(delta, epsilon, query, data, var_kernel, kernel_fun)

kde_true = true_kernel_density(data, query, kernel_fun)
print(f"true kernel density: {kde_true}")