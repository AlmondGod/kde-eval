import numpy as np
import matplotlib as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.special import erf as erf
from functools import partial as partial
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform

def gaussian_kernel(x, sigma=1.0):
    return np.exp(-0.5 * (x)**2)

def student_kernel(x, sigma=1.0):
    return 1/((x)**2+1)


datasets = {
	'covtype': [54, 581012, 10000, 2.2499],
	'shuttle': [9, 43500, 43500, 0.621882],
}

kernel_data = np.loadtxt("testing_askit.data", delimiter=',', skiprows=0)
test_data = np.loadtxt("testing_askit_query.data", delimiter=',', skiprows=0)
a = kernel_data[:, 1:]
b = test_data[:, 1:]



first_two_rows = b[5000:10000]
kernel_sums = []
kernel_sums2 = []

for i, row in enumerate(first_two_rows):
    if i % 25 == 0:
        print(i)
    
    kernel_sum = 0
    kernel_sum2 = 0
    
    for point in a:
        diff = point - row
        kernel_sum += gaussian_kernel(np.linalg.norm(diff))
        kernel_sum2 += student_kernel(np.linalg.norm(diff))
    
    kernel_sum = kernel_sum / 581012
    kernel_sum2 = kernel_sum2 / 581012
    
    kernel_sums.append(kernel_sum)
    kernel_sums2.append(kernel_sum2)

output_file = "kernel_sums_output.txt"
with open(output_file, 'a') as f:
    for kernel_sum in kernel_sums:
        f.write(str(kernel_sum) + '\n')
output_file1 = "student_sums_output.txt"
with open(output_file1, 'a') as f:
    for kernel_sum2 in kernel_sums2:
        f.write(str(kernel_sum2) + '\n')

print("Kernel sums saved to:", output_file)