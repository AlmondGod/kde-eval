import numpy as np
from scipy.special import erf as erf
from demo.eLSH import GHBE
import demo.kde_instance as kde
import time
import archived_scripts.kd_weights as kdw
import matplotlib.pyplot as plt

def gen_sample_Ts(n):
            T_values = []
            print(n)
            x = int(n / 40)
            for _ in range(40):
                T_values.append(x)
                x += int(n / 40)
            T_values.append(n - 1)
            return T_values

def gaussian_kernel(x):
    return np.exp(-(x)**2) #they use this gaussian

if __name__ == "__main__":
    # Problem Specificaiton for Gaussian Kernel
    kernel = lambda x: np.exp(-x**2)
    inverse = lambda mu: np.sqrt(-np.log(mu))
    # Creating ``Uncorrelated" instance
    num_points = 30
    clusters = 60
    scales = 3
    density = 0.01
    dimension = 30
    spread = 0.02
    Instance1 = kde.KDE_instance(kernel, inverse, num_points, density,
                     clusters, dimension, scales, spread)
    n = Instance1.N
    
    eps = 0.5
    tau = float(10**-3)
    print(f"Instance1 shape:{Instance1.X.shape}")
    hbebuild_start_time = time.time()
    kde1 = GHBE(Instance1.X, tau, eps) #hbe
    hbebuild_duration = time.time() - hbebuild_start_time
    print(f"hbe build duration: {hbebuild_duration}")

    queries = []
    hbe_estimates = []
    true_kdes = []
    hbe_times = []
    iterations = 100
    cnt = 0
    for j in range(iterations):
        # Random queries around 0
        q = np.zeros(Instance1.dimension) + np.random.randn(dimension) / np.sqrt(dimension)
        queries.append(q)
        kernel_fun = lambda x,y: np.exp(-np.linalg.norm(x-y)**2)

        kd = 0.0
        for i in range(n):
            kd = kd + kernel_fun(q, Instance1.X[:,i])
        kd = kd / n
        true_kdes.append(kd)

        start_time = time.time()
        est = kde1.AMR(q)
        duration = time.time() - start_time
        hbe_times.append(duration)
        hbe_estimates.append(est)
        print (f"Estimate: {est} True: {kd}")
        if abs((kd - est) / kd) <= eps:
            cnt = cnt + 1
        print (f"Query {j+1} rel-error: {(kd - est) / kd}")

    delta = 1 - cnt / float(iterations)
    print ("--------------------------------")
    print (f"Failure prob: {1 - cnt / float(iterations)}")
    print ("================================")

    #Random Sampling
    data = Instance1.X.transpose()
    import random_sampling as rs
    import optimal_T as ot
    #-------------------------

    rs_estimates = []
    rs_times = []

    for i, query in enumerate(queries):
        test_Ts = gen_sample_Ts(n)
        print(test_Ts)
        opt_T = ot.optimal_T(data, gaussian_kernel, q, test_Ts, delta, eps)
        print(f"optimal T: {opt_T}")

        start_time = time.time()
        # opt_T = 581012 #the above found opt T for eps = 0.1 and delta = 0.1 to be 581012 (theoretical is more than 3 times larger)
        est, _ = rs.kde_random_sampling(data, gaussian_kernel, query, opt_T, num_reps=1)
        duration = time.time() - start_time
        rs_times.append(duration)

        rs_estimates.append(est)

        kd = true_kdes[i]
        print (f"Estimate: {est} True: {kd}")
        if abs((kd - est) / kd) <= eps:
            cnt = cnt + 1
        print (f"Query {j+1} rel-error: {(kd - est) / kd}")

    delta = 1 - cnt / float(iterations)
    print ("--------------------------------")
    print (f"Failure prob: {1 - cnt / float(iterations)}")
    print ("================================")

    ginis = []
    for i, query in enumerate(queries):
        _, gini_g = kdw.kd_gini(data, gaussian_kernel, query)
        ginis.append(gini_g)

#plot for each i the accuracy of hbe, rs, and the gini of the data
plt.plot(range(iterations), true_kdes, label="True KDE")
plt.plot(range(iterations), hbe_estimates, label="HBE")
plt.plot(range(iterations), rs_estimates, label="RS")
plt.plot(range(iterations), ginis, label="Gini")
plt.legend()
plt.show()

#plot the correlation for rs and hbe between error and gini
plt.scatter(np.abs((np.array(hbe_estimates) - np.array(true_kdes))/np.array(true_kdes)), ginis, label="HBE")
plt.legend()
plt.show()

plt.scatter(np.abs((np.array(rs_estimates) - np.array(true_kdes))/np.array(true_kdes)), ginis, label="RS")
plt.legend()
plt.show()

with open("outputs/compare_algs2.txt", "a") as f:
    f.write(f"delta: {delta}, epsilon: {eps}\n")
    for i in range(iterations):
        f.write(f"Query {i+1}, true density: {true_kdes[i]}, gini: {ginis[i]} \n")
        f.write(f"hbe: {hbe_estimates[i]} with error {np.abs((hbe_estimates[i] - true_kdes[i])/true_kdes[i])} and duration: {hbe_times[i]}\n")
        f.write(f"rs: {rs_estimates[i]} with error {np.abs((rs_estimates[i] - true_kdes[i])/true_kdes[i])}  and duration: {rs_times[i]}\n")