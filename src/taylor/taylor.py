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

pairwise = np.loadtxt("outputPairwise10.txt", delimiter=' ', skiprows=0)
sumOfP = pairwise[-1]
p = pairwise[:-1]

kernel_data = np.loadtxt("covtype_normed_normalized10.csv", delimiter=',', skiprows=0)
test_data = np.loadtxt("covtype_queries_normalized10.csv", delimiter=',', skiprows=0)
a = kernel_data
b = test_data

first_two_rows = b[0:1000]
kernel_sums = []

for i, row in enumerate(first_two_rows):
    if i % 100 == 0:
        print(i)
    
    kernel_sumStudent = 0
    sP = 0
    for l in range(10):
        m = l
        while m < 10:
            sP += p[l][m]*row[m]*row[l]
            m+=1

    print("hello?")
    print(2/(1+2*10)*np.dot(row, sumOfP))
    print(sP)
    approxKernel = 581012/(1+2*(10)) - 2/(1+2*10)*np.dot(row, sumOfP) + (2/(1+2*10))**2*(sP)
    print(approxKernel)

    kernel_sums.append(kernel_sumStudent/581012)
    #kernel_sums2.append(kernel_sum2)

output_file = "studentWithnewtry.txt"
with open(output_file, 'w') as f:
    for kernel_sum in kernel_sums:
        f.write(str(kernel_sum) + '\n')

print("Kernel sums saved to:", output_file)