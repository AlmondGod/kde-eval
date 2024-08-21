import numpy as np
import matplotlib.pyplot as plt
import random_sampling as rs

#Given a delta and epsilon, find the minimum T which satisfies the problem
#constraints, which is that the estimated KDE is within $(1 + \epsilon)$-multiplicative
#away from the true estimate with less than $(1 - \delta)$ probability
def optimal_T(data, fun, query, T_values, delta, epsilon):
    kde_true = rs.kernel_density_true(data, query, fun)
    for T in T_values:
        num_true = 0
        num_false = 0
        for _ in range(100):
            mean_kde, _ = rs.kde_random_sampling(data, fun, query, T, num_reps=1)
            # print(f"mean_kde: {mean_kde}, kde_true: {kde_true}")
            if mean_kde >= (1 - epsilon) * kde_true and mean_kde <= (1 + epsilon) * kde_true:
                num_true += 1
            else: num_false += 1
        print(num_false)
        if(num_false/100 <= delta):
            return T
    return len(data) #return largest possible T (size of the data) if no trials worked

def find_optimal_Ts(data, fun, query, T_values, eval_deltas, eval_epsilons):
    optimalTs = np.zeros((len(eval_deltas), len(eval_epsilons)))
    for i, eval_delta in enumerate(eval_deltas):
        for j, eval_epsilon in enumerate(eval_epsilons):
            T = optimal_T(data, fun, query, T_values, eval_delta, eval_epsilon)
            optimalTs[i, j] = T
            print(f"delta = {eval_delta}, epsilon = {eval_epsilon}, optimal T is {T}")
    return optimalTs

# #below we compute optimal value of T for various combinations of delta and epsilon
# #we use the same data and query point as in random_sampling.py

# #delta varies between 0 and 1 and measures permitted failure rate: 
# eval_deltas = [0.99, 0.9, 0.5, 0.1, 0.01, 0.001]

# #epsilon varies infinitely positive but we want a reasonably useful margin of error:
# eval_epsilons = [1, 0.5, 0.1, 0.01, 0.001, 0.0001]

# T_values = rs.generate_sampling_points(len(rs.kernel_data))

# optimalTs_gaussian = np.zeros((len(eval_deltas), len(eval_epsilons)))
# for i, eval_delta in enumerate(eval_deltas):
#     for j, eval_epsilon in enumerate(eval_epsilons):
#         T = optimal_T(rs.kernel_data, rs.gaussian_kernel, rs.query, T_values, eval_delta, eval_epsilon)
#         optimalTs_gaussian[i, j] = T
#         print(f"delta = {eval_delta}, epsilon = {eval_epsilon}, optimal T is {T}")

# optimalTs_student = np.zeros((len(eval_deltas), len(eval_epsilons)))
# for i, eval_delta in enumerate(eval_deltas):
#     for j, eval_epsilon in enumerate(eval_epsilons):
#         T = optimal_T(rs.kernel_data, rs.student_kernel, rs.query, T_values, eval_delta, eval_epsilon)
#         optimalTs_student[i, j] = T
#         print(f"delta = {eval_delta}, epsilon = {eval_epsilon}, optimal T is {T}")

# with open("optimal_T_gaussian.txt", 'a') as f:
#     f.write(f"kde_true = {rs.kde_true_gauss}\n")
#     for i, row in enumerate(optimalTs_gaussian):
#         for j, index in enumerate(row):
#             f.write(f"for delta = {eval_deltas[i]}, epsilon = {eval_epsilons[j]}, optimal T is {index}\n")

# with open("optimal_T_student.txt", 'a') as f:
#     f.write(f"kde_true = {rs.kde_true_student}\n")
#     for i, row in enumerate(optimalTs_student):
#         for j, index in enumerate(row):
#             f.write(f"for delta = {eval_deltas[i]}, epsilon = {eval_epsilons[j]}, optimal T is {index}\n")

# #2 heatmaps of the optimal T values, one for gaussian, one for student
# #(more of a sanity check)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(optimalTs_gaussian, cmap='hot', interpolation='nearest')
# ax[0].set_title('Gaussian Kernel')
# ax[0].set_xlabel('Epsilon')
# ax[0].set_ylabel('Delta')
# ax[0].set_xticks(np.arange(len(eval_epsilons)))
# ax[0].set_xticklabels(eval_epsilons)
# ax[0].set_yticks(np.arange(len(eval_deltas)))
# ax[0].set_yticklabels(eval_deltas)

# ax[1].imshow(optimalTs_student, cmap='hot', interpolation='nearest')
# ax[1].set_title('Student Kernel')
# ax[1].set_xlabel('Epsilon')
# ax[1].set_ylabel('Delta')
# ax[1].set_xticks(np.arange(len(eval_epsilons)))
# ax[1].set_xticklabels(eval_epsilons)
# ax[1].set_yticks(np.arange(len(eval_deltas)))
# ax[1].set_yticklabels(eval_deltas)

# plt.tight_layout()
# plt.show()