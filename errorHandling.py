import numpy as np
import matplotlib as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.special import erf as erf
from functools import partial as partial
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import random

from scipy.stats import entropy


def gaussian_kernel(x, sigma=1.0):
    return np.exp(-0.5 * (x)**2)

def student_kernel(x, sigma=1.0):
    return 1/((x)**2+1)

def integrated_squared_error(p, q):
    return np.sum((p - q) ** 2)

def kullback_leibler_divergence(p, q):
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return entropy(p, q)


kernel_data = np.loadtxt("kernel_sums_output_random_testing1.txt", delimiter=',', skiprows=0)
actual_data = np.loadtxt("/Users/mtsui/kde-eval/outputs/kernel_sums_output.txt", delimiter=',', skiprows=0)
actual_data = actual_data[:3000]

data = np.genfromtxt("/Users/mtsui/kde-eval/outputs/kd_weights_gaussianTESTING.txt", delimiter=',', dtype=None, encoding=None)
data2 = np.genfromtxt("/Users/mtsui/kde-eval/outputs/kd_weights_gaussianHHI.txt", delimiter=',', dtype=None, encoding=None)


b = data
a = kernel_data
c = data2
actual_data1 = np.loadtxt("/Users/mtsui/kde-eval/outputs/kernel_sums_output.txt", delimiter=',', skiprows=0)
lolll = np.loadtxt("combined_variance.txt", delimiter=',', skiprows=1)
testing = []
for i in range(1000):
    x1 = lolll[i][1]
    x2 = actual_data1[i]
    testing.append(x1/(x2**2))
      
    
print(testing)
 
print(a)

print(b) 
b_new = np.array([row[1] for row in b])
c_new =  np.array([row[1] for row in c])

print(len(b_new))
print(b_new)

#ise = integrated_squared_error(kernel_data, actual_data)

#kld = kullback_leibler_divergence(kernel_data, actual_data)


#print("Integrated Squared Error (ISE):", ise)
#print("Kullback-Leibler Divergence (KLD):", kld)





normalized_error = np.abs(kernel_data - actual_data) / np.abs(actual_data)
correlation_normalized_error_gini = np.corrcoef(normalized_error.flatten()[:1000], testing[:1000])[0, 1]
print(correlation_normalized_error_gini)


plt.figure(figsize=(8, 6))
plt.scatter(testing, normalized_error.flatten()[:1000] , alpha=0.5)
plt.ylabel('Normalized Error')
plt.xlabel('gini coeef')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(c_new, normalized_error.flatten() , alpha=0.5)
plt.ylabel('Normalized Error')
plt.xlabel('hhi')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.hist(b_new, bins=50, alpha=0.75, edgecolor='black')
plt.xlabel('Gini Coefficient')
plt.ylabel('Frequency')
plt.title('Distribution of Gini Coefficient Values')
plt.grid(True)
plt.show()

# for i, row in enumerate(first_two_rows):
#     if i % 100 == 0:
#         print(i)
    
#     kernel_sumGauss = 0
#     kernel_sumStudent = 0
#     j = 0
#     # averageGauss = 1    #averageStudent = 1
#     # condition = 1 / (epsilon**2 * delta * averageGauss)

#     while j < 30000:
#         r = random.randint(0, 581011)
#         diff = a[r] - row
#         kernel_sumGauss += gaussian_kernel(np.linalg.norm(diff))
#         #kernel_sumStudent += student_kernel(np.linalg.norm(diff))
        
#         j += 1
#         # averageGauss = kernel_sumGauss/j
#         # print(j)
#         #averageStudent = kernel_sum/i
        
#         # condition = 2 / (epsilon**2  * averageGauss)
  
#         # print(averageGauss)
    
#         # if (averageGauss < 0.001):
#         #     output_file = "testingtestingtesting.txt"   
#         #     with open(output_file, 'a') as f:
#         #         f.write(str(i) + "space" + str(j) + '\n')
        
#     averageGauss = kernel_sumGauss/30000
#     kernel_sums.append(averageGauss)
#     #kernel_sums2.append(kernel_sum2)

# output_file = "kernel_sums_output_random_testingxdxdxdxd.txt"
# with open(output_file, 'w') as f:
#     for kernel_sum in kernel_sums:
#         f.write(str(kernel_sum) + '\n')
# # output_file1 = "student_sums_output_random.txt"
# # with open(output_file1, 'w') as f:
# #     for kernel_sum2 in kernel_sums2:
# #         f.write(str(kernel_sum2) + '\n')

# print("Kernel sums saved to:", output_file)