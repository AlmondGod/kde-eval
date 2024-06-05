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

def percent_error(actual, predicted):
    return abs((actual - predicted) / actual) * 100

# Calculate average percent error for a list of predictions
def average_percent_error(actual, predictions):
    percent_errors = [percent_error(actual[i], predictions[i]) for i in range(len(actual))]
    return sum(percent_errors) / len(percent_errors)

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


#kernel_data = np.loadtxt("kernel_sums_output_random_testing1.txt", delimiter=',', skiprows=0)
actual_data = np.loadtxt("/Users/mtsui/kde-eval/outputs/kernel_sums_output.txt", delimiter=',', skiprows=0)
actual_data = actual_data[:3000]

data = np.genfromtxt("/Users/mtsui/kde-eval/outputs/kd_weights_gaussianTESTING.txt", delimiter=',', dtype=None, encoding=None)
data2 = np.genfromtxt("/Users/mtsui/kde-eval/outputs/kd_weights_gaussianHHI.txt", delimiter=',', dtype=None, encoding=None)

cleaned_data = []
with open("/Users/mtsui/kde-eval/kernel_sums_output_random_testing_knn10000.txt", 'r') as file:
    for line in file:
        line = line.strip().strip('[]')
        cleaned_data.append([float(x) for x in line.split(',')])


knn = np.array(cleaned_data)
print(knn)
print(knn.shape)
print("hello")
b = data
#a = kernel_data
c = data2
actual_data1 = np.loadtxt("/Users/mtsui/kde-eval/outputs/kernel_sums_output.txt", delimiter=',', skiprows=0)
lolll = np.loadtxt("combined_variance.txt", delimiter=',', skiprows=1)
testing = []
testing5 = []
testing10 = []
testing50 =[]
testing100 = []
testing500 = []
testing1000 = []
testing10000 = []

for i in range(1000):
    x1 = lolll[i][1] 
    x2 = actual_data1[i]
    testing.append(x1/(x2**2))
    sum1000 = 0
    sum5 = 0
    sum10 = 0
    sum50 = 0
    sum100 = 0
    sum10000 = 0
    sum500 = 0
    for j in range(1000):
        sum1000 += (x2-gaussian_kernel(knn[i][j]))**2
    testing1000.append((sum1000 + (581012 - 1000) * (x2 - gaussian_kernel(np.linalg.norm(knn[i][999])))**2) / (581012 * x2**2))
    for j in range(1000):
        sum10000 += (x2-gaussian_kernel(knn[i][j]))**2
    testing10000.append((sum10000- + (581012 - 10000) * (x2 - gaussian_kernel(np.linalg.norm(knn[i][9999])))**2) / (581012 * x2**2))
    for k in range(500):
        sum500 += (x2-gaussian_kernel(knn[i][k]))**2
    testing500.append((sum500 + (581012 - 500) * (x2 - gaussian_kernel(np.linalg.norm(knn[i][499])))**2) / (581012 * x2**2))
    for m in range(100):
        sum100 += (x2-gaussian_kernel(knn[i][m]))**2
    testing100.append((sum100 + (581012 - 100) * (x2 - gaussian_kernel(np.linalg.norm(knn[i][99])))**2) / (581012 * x2**2))
    for l in range(50):
        sum50 += (x2-gaussian_kernel(knn[i][l]))**2
    testing50.append((sum50 + (581012 - 50) * (x2 - gaussian_kernel(np.linalg.norm(knn[i][49])))**2) / (581012 * x2**2))
   
print("Testing5:", testing5)
print("Testing10:", testing10)
print("Testing50:", testing50)
print("Testing100:", testing100)
print("Testing500:", testing500)
print("Testing1000:", testing1000)
print("kill")


with open('outputKNN1.txt', 'w') as f:
    for i in range(1000):
        # Write each row with values from all testing lists
        f.write(f"{testing[i]} {testing50[i]} {testing100[i]} {testing500[i]} {testing1000[i]}\n")
average_error_50 = average_percent_error(testing, testing50)
average_error_100 = average_percent_error(testing, testing100)
average_error_500 = average_percent_error(testing, testing500)
average_error_1000 = average_percent_error(testing, testing1000)
average_error_10000 = average_percent_error(testing, testing10000)

# Print the results
print(f"Average Percent Error for testing50: {average_error_50:.2f}%")
print(f"Average Percent Error for testing100: {average_error_100:.2f}%")
print(f"Average Percent Error for testing500: {average_error_500:.2f}%")
print(f"Average Percent Error for testing1000: {average_error_1000:.2f}%")
print(f"Average Percent Error for testing10000: {average_error_10000:.2f}%")
print("Data has been written to output.txt")