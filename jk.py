import numpy as np
import matplotlib as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.special import erf as erf
from functools import partial as partial
import sys
import numpy as np

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
first_two_rows = test_data[2000:2500]
kernel_sums = []
kernel_sums2 = []

for row in first_two_rows:
    kernel_sum = 0
    kernel_sum2 = 0
    for point in kernel_data:
        diff = point - row
        kernel_sum += gaussian_kernel(np.linalg.norm(diff))
        kernel_sum2  += student_kernel(np.linalg.norm(diff))
    kernel_sum = kernel_sum/581012
    kernel_sum2 = kernel_sum2/581012
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