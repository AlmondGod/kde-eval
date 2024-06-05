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
actual_data = np.loadtxt("kernel_sums_output.txt", delimiter=',', skiprows=0)
a = kernel_data[:, 1:]
b = test_data[:, 1:]

lolll = np.loadtxt("combined_variance.txt", delimiter=',', skiprows=1)



output_file = "actualVariance_mu.txt"

with open(output_file, 'w') as f:
    
    for i, row in enumerate (actual_data):
        x1 = lolll[i][1]
        x2 = actual_data[i]

        print(x1)
        print(x2)
    
        f.write(f"{x1/x2},{x1/(x2**2)}\n")
  


first_two_rows = b[0:1000]
kernel_sums = []
kernel_sums2 = []

for i, row in enumerate(first_two_rows):
    if i % 25 == 0:
        print(i)
    
    kernel_sum = 0
    kernel_sum2 = 0
    
    for point in a:
        diff = point - row
        kernel_sum += gaussian_kernel(np.linalg.norm(diff))**2
        kernel_sum2 += (actual_data[i]-gaussian_kernel(np.linalg.norm(diff)))**2
    
    kernel_sum = kernel_sum / 581012
    kernel_sum2 = kernel_sum2 / 581012
    
    kernel_sums.append(kernel_sum)
    kernel_sums2.append(kernel_sum2)

percent_errors = []
for k1, k2 in zip(kernel_sums, kernel_sums2):
    if k2 != 0:
        percent_error = ((k2 - k1) / k2) * 100
    else:
        percent_error = float('inf')  # Handle division by zero
    percent_errors.append(percent_error)

# Calculate the average percent error
if percent_errors:
    average_percent_error = sum(percent_errors) / len(percent_errors)
else:
    average_percent_error = 0


output_file = "combined_variance.txt"

with open(output_file, 'w') as f:
    f.write("Actual var, Bound, Percent Error\n")  
    for k1, k2 in zip(kernel_sums, kernel_sums2):
        percent_error = ((k2 - k1) / k2) * 100 if k2 != 0 else float('inf')  
        f.write(f"{k1},{k2},{percent_error:.2f}\n")
    f.write(f"\nAverage Percent Error: {average_percent_error:.2f}\n")



print("Kernel sums saved to:", output_file)