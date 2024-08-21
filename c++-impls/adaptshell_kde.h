#ifndef ADAPTIVE_KDE_H
#define ADAPTIVE_KDE_H

#include <vector>
#include <string>

// Forward declaration of nanoflann's PointCloud
namespace nanoflann {
    template <typename T>
    struct PointCloud;
}

// PointCloud structure
template <typename T>
struct PointCloud {
    std::vector<std::vector<T>> pts;

    inline size_t kdtree_get_point_count() const;
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const;
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& bb) const;
};

// Function declarations
std::vector<double> kdProjection(const std::vector<std::vector<double>>& X, const std::vector<double>& z);
double kernel(const std::vector<double>& u);
double kernel(const double dist);
double kernelSquared(const std::vector<double>& u);
double estimateKernelSquared(const PointCloud<double>& cloud, const std::vector<double>& query);
double adaptiveKDE(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double variance_estimate, double epsilon);
double kernelDensityEstimation(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double epsilon);
double trueKernelDensity(const std::vector<std::vector<double>>& X, const std::vector<double>& z);
std::vector<std::vector<double>> readDataset(const std::string& filename);

#endif // ADAPTIVE_KDE_H