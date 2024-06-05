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
epsilon = 0.2
delta = 0.1
k =  math.ceil(math.log(1/delta, 3/2))
print(k)
 


first_two_rows = b[0:3000]
kernel_sums = []
kernel_sums2 = []

for i, row in enumerate(first_two_rows):
    if i % 100 == 0:
        print(i)
    
    kernel_sumGauss = 0
    kernel_sumStudent = 0
    j = 0
    # averageGauss = 1    #averageStudent = 1
    # condition = 1 / (epsilon**2 * delta * averageGauss)

    while j < 30000:
        r = random.randint(0, 581011)
        diff = a[r] - row
        kernel_sumGauss += gaussian_kernel(np.linalg.norm(diff))
        #kernel_sumStudent += student_kernel(np.linalg.norm(diff))
        
        j += 1
        # averageGauss = kernel_sumGauss/j
        # print(j)
        #averageStudent = kernel_sum/i
        
        # condition = 2 / (epsilon**2  * averageGauss)
  
        # print(averageGauss)
        # run nearest neighbor search, boostrap for upperbound on variance
        # if (averageGauss < 0.001):
        #     output_file = "testingtestingtesting.txt"   
        #     with open(output_file, 'a') as f:
        #         f.write(str(i) + "space" + str(j) + '\n')
        
    averageGauss = kernel_sumGauss/30000
    kernel_sums.append(averageGauss)
    #kernel_sums2.append(kernel_sum2)

output_file = "kernel_sums_output_random_testing1.txt"
with open(output_file, 'w') as f:
    for kernel_sum in kernel_sums:
        f.write(str(kernel_sum) + '\n')
# output_file1 = "student_sums_output_random.txt"
# with open(output_file1, 'w') as f:
#     for kernel_sum2 in kernel_sums2:
#         f.write(str(kernel_sum2) + '\n')

print("Kernel sums saved to:", output_file)