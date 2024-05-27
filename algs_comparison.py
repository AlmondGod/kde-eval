import demo.rehashing as rehashing
# import matthew_random_sample as mrs
import random_sampling as rs
import numpy as np
import optimal_T as ot
import time

#compare hashing based estimator from rehashing, random sampling, and matthew random sampling
#on the same dataset, querypoint, delta allowed percent error and epsilon multiplicative threshold 
#error allowed

data = np.loadtxt(open("C:/users/almon/kde-eval/data/our_data_askit.data", "rb"), delimiter=",",\
                        skiprows=0)[:, 1:]
points = data.transpose() #for their hbe they use the transposed data matrix
(d,n) = points.shape
    
delta = 0.1
epsilon = 0.1
num_reps = 100

#suggested value is 0.1 / sqrt{n} (they use 0.0001 in their example, this comes out to about that)
tau = 0.1 / np.sqrt(n)
print(tau)

weights = np.ones(n) / n

gaussian_fun = lambda x,y: np.exp(-0.5 * np.linalg.norm(x-y)**2)

hbe = rehashing.rehashing(points, weights, gaussian_fun)
hbe.create_sketch(method='random', accuracy=epsilon, threshold=tau,\
    num_of_means=1)

i_s = [7777, 8888, 9999]
for i in i_s:
    query = rs.query_data[i]

    true_density_them = hbe.eval_density(query)
    print(f"true density them:{true_density_them}")
    true_density_us = rs.kernel_density_true(data, query, rs.gaussian_kernel)
    print(f"true density us:{true_density_us}")

    hbe_start_time = time.time()
    hbe_density = hbe.eval_sketch(query)
    hbe_duration = time.time() - hbe_start_time
    hbe_error = np.abs(hbe_density - true_density_them) / true_density_them
    print(f"hbe density: {hbe_density} with duration {hbe_duration}\n and percent error {hbe_error}\n")

    includet_rs_start_time = time.time()

    def gen_sample_Ts(n):
        T_values = []
        print(n)
        x = int(n / 40)
        for _ in range(40):
            T_values.append(x)
            x += int(n / 40)
        T_values.append(n - 1)
        return T_values
    test_Ts = gen_sample_Ts(n)
    print(test_Ts)
    opt_T = int(ot.find_optimal_Ts(data, rs.gaussian_kernel, query, test_Ts, [delta], [epsilon])[0][0])
    print(f"optimal T: {opt_T}")

    rs_start_time = time.time()
    # opt_T = 581012 #the above found opt T for eps = 0.1 and delta = 0.1 to be 581012 (theoretical is more than 3 times larger)
    rs_density, _ = rs.kde_random_sampling(data, rs.gaussian_kernel, query, opt_T, num_reps=1)
    rs_duration = time.time() - rs_start_time
    rs_total_duration = time.time() - includet_rs_start_time
    rs_error = np.abs(rs_density - true_density_us) / true_density_us
    print(f"rs_density: {rs_density} with duration {rs_duration} (total {rs_total_duration}) and percent error {rs_error}\n")

    with open("outputs/compare_algs.txt", 'a') as f:
        f.write(f"query point: query_data[{i}]\n")
        f.write(f"true density: {true_density_them}, {true_density_us}\n")
        f.write(f"hbe density: {hbe_density} with duration {hbe_duration}s and percent error {hbe_error}\n")
        f.write(f"rs_density (with T = {opt_T}): {rs_density} with duration {rs_duration}s (total {rs_total_duration}) and percent error {rs_error}\n\n")