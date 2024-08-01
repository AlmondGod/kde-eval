#include "adaptshell_kde.h"
#include "nanoflann.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>

using namespace nanoflann;

template <typename T>
struct PointCloud
{
    std::vector<std::vector<T> > pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return pts[idx][dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

// Function declarations
std::vector<double> kdProjection(const std::vector<std::vector<double>>& X, const std::vector<double>& z);
double kernelSquared(const std::vector<double>& u);
double estimateKernelSquared(const PointCloud<double>& cloud, const std::vector<double>& query);
double adaptiveKDE(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double variance_estimate, double epsilon);

// Main KDE function
double kernelDensityEstimation(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double epsilon) {
    PointCloud<double> cloud;
    cloud.pts = X;
    double variance_estimate = estimateKernelSquared(cloud, z);
    std::cout << "Estimated variance: " << variance_estimate << std::endl;
    return adaptiveKDE(X, z, variance_estimate, epsilon);
}

// K-d projection function
std::vector<double> kdProjection(const std::vector<std::vector<double>>& X, const std::vector<double>& z) {
    std::vector<double> projections;
    projections.reserve(X.size());
    
    for (const auto& x : X) {
        double projection = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            projection += (x[i] - z[i]) * (x[i] - z[i]);
        }
        projections.push_back(std::sqrt(projection));
    }
    
    return projections;
}

// Kernel function
double kernel(const std::vector<double>& u) {
    double result = 1.0;
    for (double ui : u) {
        result *= std::exp(-0.5 * ui * ui) / std::sqrt(2 * M_PI);
    }
    return result;
}

double kernel(const double dist) {
    double result = 1.0;
    result = std::exp(-0.5 * dist) / std::sqrt(2 * M_PI);
    return result;
}

// Kernel squared function
double kernelSquared(const std::vector<double>& u) {
    double result = 1.0;
    for (double ui : u) {
        result *= std::exp(-0.5 * ui * ui) / std::sqrt(2 * M_PI);
    }
    return result * result;
}

double estimateKernelSquared(const PointCloud<double>& cloud, const std::vector<double>& query) {
    typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud<double> >, PointCloud<double>, -1> my_kd_tree_t;

    int dim = query.size();
    my_kd_tree_t index(dim, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();

    size_t dataset_size = cloud.pts.size();
    std::cout << "Dataset size: " << dataset_size << std::endl;

    double closest_distance = 0.0;
    size_t k = 1;
    while (closest_distance == 0.0 && k <= dataset_size) {
        std::vector<size_t> ret_index(k);
        std::vector<double> out_dist_sqr(k);
        nanoflann::KNNResultSet<double> resultSet(k);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        index.findNeighbors(resultSet, &query[0], nanoflann::SearchParameters());
        
        for (size_t i = 0; i < k; ++i) {
            std::cout << "k=" << k << ", distance^2: " << out_dist_sqr[i] << std::endl;
            if (out_dist_sqr[i] > 0.0) {
                closest_distance = std::sqrt(out_dist_sqr[i]);
                std::cout << "Found non-zero distance at k=" << k << ", distance: " << closest_distance << std::endl;
                break;
            }
        }
        
        if (closest_distance == 0.0) {
            k++;
        }
    }

    if (closest_distance == 0.0) {
        std::cout << "Warning: Could not find a non-zero distance." << std::endl;
        return 0.0;
    }

    double kernel_sq_estimate = 0;
    size_t points_counted = 0;
    double current_radius = closest_distance;
    size_t total_points = 0;
    int iters = 0;
    double closest_radius = current_radius;

    while (total_points < dataset_size) {
        std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
        nanoflann::SearchParameters params;
        size_t count = index.radiusSearch(&query[0], current_radius*current_radius, ret_matches, params);
        std::cout << "Iteration " << iters << ": count=" << count << ", current_radius=" << current_radius << std::endl;
        size_t new_points = count - total_points;
        std::cout << "New points: " << new_points << ", closest_radius: " << closest_radius << ", current_radius: " << current_radius << std::endl;
        if (new_points > 0) {
            // std::vector<double> u(dim, closest_radius);
            double k = kernel(closest_radius);
            double k_sq = k * k;
            std::cout << "k: " << k << ", k_sq: " << k_sq << std::endl;
            kernel_sq_estimate += new_points * k_sq;
            points_counted += new_points;
            std::cout << "Current kernel_sq_estimate: " << kernel_sq_estimate << ", points_counted: " << points_counted << std::endl;
        }
        
        current_radius *= 2;
        total_points = count;
        if (iters > 0) {
            closest_radius *= 2;
        }

        iters++;
    }

    std::cout << "Final kernel_sq_estimate: " << kernel_sq_estimate << ", points_counted: " << points_counted << std::endl;

    return points_counted > 0 ? kernel_sq_estimate / points_counted : 0;
}

// Adaptive KDE function
double adaptiveKDE(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double variance_estimate, double epsilon) {
    double n = X.size();
    double d = z.size();  // Dimensionality of the data

    // Set up random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, n - 1);

    int j = 1;
    int r = 0;
    double average = 0;
    double kernel_sum;
    std::vector<double> diff(d);
    while (average == 0) {
        r = distribution(generator);
        for (size_t i = 0; i < d; ++i) {
            diff[i] = X[r][i] - z[i];
        }
        std::cout << "Initial diff: " << diff[0] << ", " << diff[1] << ", " << diff[2] << ", " << diff[3] << std::endl;
        std::cout << "Initial kernel diff: " << kernel(diff) << std::endl;
        kernel_sum = kernel(diff);
        average = kernel_sum;
    }
        
    double t = 2 * variance_estimate / (std::pow(average, 2) * std::pow(epsilon, 2));

    std::cout << "Initial kernel sum: " << average << std::endl;
    std::cout << "Initial T: " << t << std::endl;
    while (j < t && j < n) {
        r = distribution(generator);
        for (size_t i = 0; i < d; ++i) {
            diff[i] = X[r][i] - z[i];
        }
        kernel_sum += kernel(diff);
        // std::cout << "kernel diff: " << kernel(diff) << std::endl;
        j++;
        average = kernel_sum / j;
        t = 2 * variance_estimate / (std::pow(epsilon, 2) * std::pow(average, 2));
        if (j % 1000 == 0) {
            std::cout << "j: " << j << ", t: " << t << std::endl;
        }
    }

    std::cout << "Final kernel estimate: " << average << std::endl;
    std::cout << "Number of samples (j): " << j << std::endl;

    // Calculate actual kernel (for comparison)
    double actual_kernel = 0.0;
    for (const auto& x : X) {
        for (size_t i = 0; i < d; ++i) {
            diff[i] = x[i] - z[i];
        }
        actual_kernel += kernel(diff);
    }
    actual_kernel /= n;

    double percent_error = std::abs(average - actual_kernel) / actual_kernel;
    std::cout << "Actual kernel, Percent error: " << actual_kernel << ", " << percent_error << std::endl;

    return average;
}

double trueKernelDensity(const std::vector<std::vector<double>>& X, const std::vector<double>& z) {
    double n = X.size();
    double d = X[0].size();
    double h = std::pow(n, -1.0 / (d + 4));
    
    double density = 0.0;
    for (const auto& x : X) {
        std::vector<double> diff(d);
        for (size_t i = 0; i < d; ++i) {
            diff[i] = (x[i] - z[i]) / h;
        }
        density += kernel(diff);
    }
    
    return density / (n * std::pow(h, d));
}

std::vector<std::vector<double>> readDataset(const std::string& filename) {
    std::vector<std::vector<double>> dataset;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        dataset.push_back(row);
    }
    
    return dataset;
}

int main() {
    // Read the dataset
    std::string filename = "test.txt";
    std::vector<std::vector<double>> dataset = readDataset(filename);
    
    if (dataset.empty()) {
        std::cerr << "Failed to read the dataset or the dataset is empty." << std::endl;
        return 1;
    }
    
    // Select a random datapoint as query
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, dataset.size() - 1);
    int randomIndex = distribution(generator);
    std::vector<double> query = dataset[randomIndex];

    std::cout << "query: " << query[0] << ", " << query[1] << "," << query[2] << "," << query[3] << std::endl;
    
    // Remove the query point from the dataset
    dataset.erase(dataset.begin() + randomIndex);
    
    // Set epsilon
    double epsilon = 0.1;  // You can adjust this value
    
    // Run kernel density estimation
    double adaptiveDensity = kernelDensityEstimation(dataset, query, epsilon);

    // Compute true kernel density
    double trueDensity = trueKernelDensity(dataset, query);
    
    // Compute percent error
    double percentError = (trueDensity - adaptiveDensity) / trueDensity * 100;
    
    // Output results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Kernel Density Estimation results:" << std::endl;
    std::cout << "Query point index: " << randomIndex << std::endl;
    std::cout << "Number of samples: " << dataset.size() << std::endl;
    std::cout << "Adaptive KDE estimate: " << adaptiveDensity << std::endl;
    std::cout << "True kernel density: " << trueDensity << std::endl;
    std::cout << "Percent error: " << percentError << "%" << std::endl;
    
    return 0;
}