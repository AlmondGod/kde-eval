#given a kernel function, a dataset, and a query point, we cant to calculate a metric
#which is higher if there are a few datapoints contributing massively to the kernel density
#and lower if there are many datapoints contributing equally to the kernel density
import numpy as np
import matplotlib.pyplot as plt
import random_sampling as rs

def kd_weights(data, fun, query):
    kde_estimates = []
    for i in range(len(data)):
        diff = query - data[i]
        kde_estimates.append(fun(np.linalg.norm(diff)))
    return kde_estimates

def kd_entropy(data, fun, query):
    kde_estimates = kd_weights(data, fun, query)
    weights = np.array(kde_estimates)
    weights = weights / np.sum(weights)
    log_weights = []
    for i in range(len(weights)):
        if weights[i] == 0: log_weights.append(0)
        else: log_weights.append(np.log(weights[i])) #i did this cause dividing by 0 was an issue
    entropy = -np.sum(weights * log_weights)
    entropy = entropy / np.log(len(weights)) #normalized entropy to 0 to 1 scale
    return weights, entropy

def kd_basic_gini(data, fun, query):
    kde_estimates = kd_weights(data, fun, query)
    weights = np.array(kde_estimates)
    weights = weights / np.sum(weights)
    gini = 1 - np.sum(weights**2)
    return weights, gini

def kd_gini(data, fun, query):
    kde_estimates = kd_weights(data, fun, query)
    weights = np.array(kde_estimates)
    weights = weights / np.sum(weights)
    sorted_weights = np.sort(weights)
    n = len(weights)
    cumulative_weights = np.cumsum(sorted_weights)
    sum_weights = cumulative_weights[-1]
    lorenz_curve = cumulative_weights / sum_weights
    lorenz_curve = np.insert(lorenz_curve, 0, 0)  # Ensure the curve starts at (0,0)
    B = np.trapz(lorenz_curve, dx=1/n)
    gini = 1 - 2 * B
    return weights, gini

def kd_hhi(data, fun, query):
    kde_estimates = kd_weights(data, fun, query)
    weights = np.array(kde_estimates)
    weights = weights / np.sum(weights)
    hhi = np.sum(weights**2)
    return weights, hhi

queries = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
for i in queries:
    query = rs.query_data[i]
    weights_g, entropy_g = kd_entropy(rs.kernel_data, rs.gaussian_kernel, query)
    weights_s, entropy_s = kd_entropy(rs.kernel_data, rs.student_kernel, query)
    print(f"Gaussian Entropy:{entropy_g}, Student Entropy:{entropy_s}")

    weights_g, gini_g = kd_gini(rs.kernel_data, rs.gaussian_kernel, query)
    weights_s, gini_s = kd_gini(rs.kernel_data, rs.student_kernel, query)
    print(f"Gaussian Gini:{gini_g}, Student Gini:{gini_s}")

    weights_g, hhi_g = kd_hhi(rs.kernel_data, rs.gaussian_kernel, query)
    weights_s, hhi_s = kd_hhi(rs.kernel_data, rs.student_kernel, query)
    print(f"Gaussian HHI:{hhi_g}, Student HHI:{hhi_s}")

    with open("outputs/kd_weights_gaussian.txt", 'a') as f:
        f.write(f"query_data[{i}], {gini_g}, {entropy_g}. {hhi_g}\n")
        
    with open("outputs/kd_weights_student.txt", 'a') as f:
        f.write(f"query_data[{i}], {gini_s}, {entropy_s}. {hhi_s}\n")

    # # plot the two weight distributions
    # fig, axs = plt.subplots(2)
    # axs[0].plot(weights_g)
    # axs[0].set_title('Gaussian Kernel Weights')
    # axs[1].plot(weights_s)
    # axs[1].set_title('Student-t Kernel Weights')

    # plt.show()