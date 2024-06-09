"""
comparing kexin/paris hbe with our new proposed algorithm 
(hbe to estimate variance for T estimate with adaptive random sampling)
on the askit data and query datasets
"""

from demo.eLSH import GHBE
from matthew_random_sample import a, b
import numpy as np
import random
import time
import demo.rehashing as rehashing


def gaussian_kernel(x, sigma=1.0):
    return np.exp(-0.5 * (x)**2)

def true_kernel_density(data, query, kernel_fun):
    kernel_sum = 0
    for i in range(len(data)):
        kernel_sum += kernel_fun(data[i], query)
    return kernel_sum / len(data)

def new_alg (epsilon, query, data, var_kernel, kernel_fun):
    tau = 0.1 / np.sqrt(len(data))
    print(f"tau: {tau}")
    points = np.array(data).transpose()
    print(f"points shape: {points.shape[1]}")
    ghbe = GHBE(points, tau, eps=10, kernel_fun=var_kernel)
    var_est = ghbe.AMR(query)
    print(f"variance estimate: {var_est}")
    true_var = true_kernel_density(data, query, var_kernel)
    print(f"true variance: {true_var}")

    j = 0
    averageGauss = 1   
    kernel_sumGauss = 0

    while j < len(data):
        r = random.randint(0, len(data) - 1)
        kernel_sumGauss += kernel_fun(data[r], query)
        
        j += 1
        averageGauss = kernel_sumGauss/j

        T = 2 * var_est / (epsilon**2 * averageGauss)
        print(f"j: {j}, current estimate: {averageGauss}, current T: {T}")
        if (j > T):
            break
        
    averageGauss = kernel_sumGauss/30000

    return averageGauss

def new_alg_preprocessed(query, data, kernel_fun, hbe):
    var = hbe.AMR(query)
    j = 0
    average = 1
    kernel_sum = 0

    while j < len(data):
        r = random.randint(0, len(data) - 1)
        kernel_sum += kernel_fun(data[r], query)
        
        j += 1
        average = kernel_sum/j

        T = 2 * var / (epsilon**2 * average)
        if (j > T):
            print(f"LAST J: {j}, LAST T: {T}")
            break
    
    return average, j

def new_alg_w_rehashing(query, data, epsilon, kernel_fun):
    points = data.transpose() #for their hbe they use the transposed data matrix
    (d,n) = points.shape
    tau = 0.1 / np.sqrt(n)
    weights = np.ones(n) / n
    sketch = rehashing.rehashing(points, weights, kernel_fun)
    sketch.create_sketch(method='ska', accuracy=epsilon, threshold=tau, num_of_means=1)
    var = sketch.eval_sketch(query)
    j = 0
    average = 1
    kernel_sum = 0

    while j < len(data):
        r = random.randint(0, len(data) - 1)
        kernel_sum += kernel_fun(data[r], query)
        
        j += 1
        average = kernel_sum/j

        T = 2 * var / (epsilon**2 * average)
        print(T)
        if (j > T):
            break

    return average

var_kernel = lambda x,y: np.exp(-1 * np.linalg.norm(x-y)**2)
kernel_fun = lambda x,y: np.exp(-0.5 * np.linalg.norm(x-y)**2)

# data = np.genfromtxt('data/optdigits.csv', delimiter=',')

data = a[random.sample(range(len(a)), 1000)]
print(data[0])
# delta = 0.1
epsilon = 0.5

#initialize hbe for new alg
new_inst_start = time.time()
ghbe = GHBE(np.array(data).transpose(), 0.1 / np.sqrt(len(data)), 1, kernel_fun=var_kernel)   
new_inst_time = time.time() - new_inst_start

#Initialize vanilla hbe
points = np.array(data).transpose()
tau = 0.1 / np.sqrt(len(points))
hbe_inst_start = time.time()
vhbe = GHBE(points, tau, epsilon, kernel_fun=kernel_fun)
hbe_inst_time = time.time() - hbe_inst_start

trues = []
true_vars = []
vars = []
new_alg_estimates = []
hbe_estimates = []
hbe_durations = []
new_durations = []
new_inst_times = []
hbe_inst_times = []
Ts = []
qs = [1, 2345, 3438, 2834, 9847, 5839, 3879, 3231, 8032, 5031]
for q in qs: 
    query = b[q]
    #set data to be 1000 random samples from a where a is a 2d array, so get RANDOM samples
    
    # #ska rehashing
    # sr_start = time.time()
    # points = data.transpose()
    # (d,n) = points.shape
    # tau = 0.1 / np.sqrt(n)
    # weights = np.ones(n) / n
    # gaussian_fun = lambda x,y: np.exp(-0.5 * np.linalg.norm(x-y)**2)
    # rehash = rehashing.rehashing(points, weights, kernel_fun)
    # rehash.create_sketch(method='ska', accuracy=epsilon, threshold= tau, num_of_means=1)
    # sr_estimate = rehash.eval_sketch(query)
    # sr_duration = time.time() - sr_start

    # #rehashing new alg
    # rehashing_start = time.time()
    # rehashing_estimate = new_alg_w_rehashing(query, data, epsilon, kernel_fun)
    # rehashing_duration = time.time() - rehashing_start

    kde_true = true_kernel_density(data, query, kernel_fun)
    #algo with hashing based estimator for variance estimation
    new_start = time.time()
    kd_estimate, T = new_alg_preprocessed(query, data, kernel_fun, ghbe)
    var = ghbe.AMR(query)
    true_var = true_kernel_density(data, query, var_kernel)
    print(f"var: {var}, true var: {true_var}, percent error: {abs(var - true_var) / true_var}")
    # kd_estimate = new_alg(delta, epsilon, query, data, var_kernel, kernel_fun)
    new_duration = time.time() - new_start
    print(f"true kernel density: {kde_true},\n"
        + f"new alg estimate: {kd_estimate}, percent error: {abs(kde_true - kd_estimate) / kde_true}, instantiation time: {new_inst_time}, query duration: {new_duration} \n")

    #vanilla hashing based estimator
    hbe_start = time.time()
    hbe_est = vhbe.AMR(query)
    hbe_duration = time.time() - hbe_start

    hbe_durations.append(hbe_duration)
    hbe_inst_times.append(hbe_inst_time)
    hbe_estimates.append(hbe_est)
    trues.append(kde_true)
    new_inst_times.append(new_inst_time)
    new_alg_estimates.append(kd_estimate)
    true_vars.append(true_var)
    vars.append(var)
    new_durations.append(new_duration)
    Ts.append(T)
    # with open("outputs/new_alg_eval.txt", "a") as f:
    #     f.write(f"on query {q}, var: {var}, true var: {true_var}, percent error: {(var - true_var) / true_var}\n"
    #         + f"true kernel density: {kde_true},\n"
    #             + f"new alg estimate: {kd_estimate}, percent error: {abs(kde_true - kd_estimate) / kde_true}, instantiation time: {new_inst_time}, query duration: {new_duration},\n"
    #             + f"hbe estimate: {hbe_est}, percent error: {abs(kde_true - hbe_est) / kde_true}, instantiation time: {hbe_inst_time}, query duration: {hbe_duration} \n\n")
                # + f"rehashing estimate: {rehashing_estimate}, percent error: {abs(kde_true - rehashing_estimate) / kde_true}, query duration: {rehashing_duration} \n\n"
                # + f"straight rehashing: {sr_estimate}, percent error: {abs(kde_true - sr_estimate) / kde_true}, query duration: {sr_duration} \n\n")

with open("durations_accuracies.txt", "a") as f:
    f.write(f"new alg durations: {new_durations},\n"
        + f"hbe durations: {hbe_durations},\n"
        + f"new alg instantiation times: {new_inst_times},\n"
        + f"hbe instantiation times: {hbe_inst_times},\n"
        + f"true kernel densities: {trues},\n"
        + f"new alg estimates: {new_alg_estimates},\n"
        + f"hbe estimates: {hbe_estimates},\n"
        + f"new alg errors: {[abs(trues[i] - new_alg_estimates[i]) / trues[i] for i in range(len(trues))]},\n"
        + f"hbe errors: {[abs(trues[i] - hbe_estimates[i]) / trues[i] for i in range(len(trues))]},\n"
        + f"true variances: {true_vars},\n"
        + f"variances: {vars},\n"
        + f"Ts: {Ts},\n")
