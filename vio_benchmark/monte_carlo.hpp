// monte_carlo.hpp
#pragma once
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include "data_io.hpp"
#include "benchmark_metrics.hpp"

// Configuration for Monte Carlo
struct MonteCarloConfig {
    int num_runs = 30;              // Minimum for statistical significance
    int base_seed = 12345;
    double warmup_seconds = 2.0;    // Skip first 2 seconds for convergence
    std::vector<std::string> sequences = {"MH01", "MH02", "MH03", "MH04", "MH05"};
};

struct MonteCarloResult {
    std::string filter_name;
    std::string sequence;
    
    // Position RMSE statistics
    double pos_rmse_mean;
    double pos_rmse_std;
    double pos_rmse_min;
    double pos_rmse_max;
    std::vector<double> pos_rmse_all;
    
    // Orientation RMSE statistics
    double ori_rmse_mean;
    double ori_rmse_std;
    double ori_rmse_min;
    double ori_rmse_max;
    std::vector<double> ori_rmse_all;
    
    // Timing statistics
    double time_ms_mean;
    double time_ms_std;
    std::vector<double> time_ms_all;
    
    // NEES statistics
    double nees_mean;
    double nees_std;
    std::vector<double> nees_all;
    
    // 95% confidence intervals
    double pos_rmse_ci_low;
    double pos_rmse_ci_high;
    double ori_rmse_ci_low;
    double ori_rmse_ci_high;
};

// Compute mean with confidence interval
inline void compute_ci(const std::vector<double>& data, double& mean, double& ci_low, double& ci_high) {
    int n = data.size();
    if (n < 2) {
        mean = ci_low = ci_high = 0;
        return;
    }
    
    mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    
    // Standard deviation
    double sq_sum = 0;
    for (double x : data) sq_sum += (x - mean) * (x - mean);
    double std_dev = std::sqrt(sq_sum / (n - 1));
    
    // 95% confidence interval using t-distribution (approximation: 2 * SE)
    double se = std_dev / std::sqrt(n);
    ci_low = mean - 1.96 * se;
    ci_high = mean + 1.96 * se;
}

// Compute Cohen's d effect size
inline double cohens_d(const std::vector<double>& a, const std::vector<double>& b) {
    double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
    double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / b.size();
    
    double var_a = 0, var_b = 0;
    for (double x : a) var_a += (x - mean_a) * (x - mean_a);
    for (double x : b) var_b += (x - mean_b) * (x - mean_b);
    var_a /= (a.size() - 1);
    var_b /= (b.size() - 1);
    
    double pooled_sd = std::sqrt((var_a + var_b) / 2);
    return std::abs(mean_a - mean_b) / pooled_sd;
}

// Paired t-test
inline double paired_t_test(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.size() < 2) return 1.0;
    
    std::vector<double> diff;
    for (size_t i = 0; i < a.size(); i++) {
        diff.push_back(a[i] - b[i]);
    }
    
    double mean_diff = std::accumulate(diff.begin(), diff.end(), 0.0) / diff.size();
    double var_diff = 0;
    for (double d : diff) var_diff += (d - mean_diff) * (d - mean_diff);
    var_diff /= (diff.size() - 1);
    
    double t_stat = mean_diff / std::sqrt(var_diff / diff.size());
    
    // Approximate p-value for two-tailed test (simplified)
    // In production, use a proper statistics library
    double p_value = 2.0 * std::exp(-0.5 * t_stat * t_stat);  // Rough approximation
    return std::min(p_value, 1.0);
}

// Main Monte Carlo runner (to be expanded with actual filter calls)
template<typename FilterFunc>
MonteCarloResult run_monte_carlo(
    FilterFunc filter,
    const std::string& filter_name,
    const std::string& sequence,
    const MonteCarloConfig& config,
    const std::vector<std::string>& imu_paths,
    const std::vector<std::string>& gt_paths
) {
    MonteCarloResult result;
    result.filter_name = filter_name;
    result.sequence = sequence;
    
    std::vector<double> pos_rmses, ori_rmses, times, neeses;
    
    for (int run = 0; run < config.num_runs; run++) {
  // Create a fresh RNG for this run with unique seed
    RNG rng(config.base_seed + run);
    
    // Load data
    AlignedData ad = load_and_align_data(...);
    
    // Add noise with this run's RNG
    add_noise_to_measurements(ad, rng, ...);
    
    // Run filter with this RNG
    FilterResult fr = filter(ad, rng);
        printf("\r  Monte Carlo %s: run %d/%d", filter_name.c_str(), run + 1, config.num_runs);
        fflush(stdout);
        
        // Load data with new seed for noise injection
        Dataset ds = load_euroc(imu_paths[0], gt_paths[0]);  // Simplified
        AlignedData ad = align_dataset(ds);
        
        // Create RNG with run-specific seed
        RNG rng(config.base_seed + run);
        
        // Run filter
        FilterResult fr = filter(ad, rng);
        
        pos_rmses.push_back(fr.pos_rmse);
        ori_rmses.push_back(fr.ori_rmse_deg);
        times.push_back(fr.avg_time_ms);
        
        // TODO: Store NEES (will implement later)
        neeses.push_back(1.0);  // Placeholder
    }
    printf("\n");
    
    // Compute statistics
    result.pos_rmse_all = pos_rmses;
    result.ori_rmse_all = ori_rmses;
    result.time_ms_all = times;
    result.nees_all = neeses;
    
    compute_ci(pos_rmses, result.pos_rmse_mean, result.pos_rmse_ci_low, result.pos_rmse_ci_high);
    compute_ci(ori_rmses, result.ori_rmse_mean, result.ori_rmse_ci_low, result.ori_rmse_ci_high);
    
    // Standard deviations
    double sq_sum_pos = 0, sq_sum_ori = 0, sq_sum_time = 0;
    for (double x : pos_rmses) sq_sum_pos += (x - result.pos_rmse_mean) * (x - result.pos_rmse_mean);
    for (double x : ori_rmses) sq_sum_ori += (x - result.ori_rmse_mean) * (x - result.ori_rmse_mean);
    for (double x : times) sq_sum_time += (x - result.time_ms_mean) * (x - result.time_ms_mean);
    
    result.pos_rmse_std = std::sqrt(sq_sum_pos / (config.num_runs - 1));
    result.ori_rmse_std = std::sqrt(sq_sum_ori / (config.num_runs - 1));
    result.time_ms_std = std::sqrt(sq_sum_time / (config.num_runs - 1));
    
    result.pos_rmse_min = *std::min_element(pos_rmses.begin(), pos_rmses.end());
    result.pos_rmse_max = *std::max_element(pos_rmses.begin(), pos_rmses.end());
    result.ori_rmse_min = *std::min_element(ori_rmses.begin(), ori_rmses.end());
    result.ori_rmse_max = *std::max_element(ori_rmses.begin(), ori_rmses.end());
    
    return result;
}

// Print statistical comparison
inline void print_statistical_comparison(
    const MonteCarloResult& eskf,
    const MonteCarloResult& ukf,
    const MonteCarloResult& hybrid
) {
    printf("\n");
    printf("================================================================================\n");
    printf("  STATISTICAL SIGNIFICANCE ANALYSIS\n");
    printf("================================================================================\n");
    printf("\n");
    
    // Position RMSE comparison
    double p_eskf_ukf = paired_t_test(eskf.pos_rmse_all, ukf.pos_rmse_all);
    double p_eskf_hybrid = paired_t_test(eskf.pos_rmse_all, hybrid.pos_rmse_all);
    double p_ukf_hybrid = paired_t_test(ukf.pos_rmse_all, hybrid.pos_rmse_all);
    
    double d_eskf_ukf = cohens_d(eskf.pos_rmse_all, ukf.pos_rmse_all);
    double d_eskf_hybrid = cohens_d(eskf.pos_rmse_all, hybrid.pos_rmse_all);
    double d_ukf_hybrid = cohens_d(ukf.pos_rmse_all, hybrid.pos_rmse_all);
    
    printf("Position RMSE Statistical Comparison:\n");
    printf("  ESKF vs UKF:   p = %.6f  (Cohen's d = %.2f)  %s\n", 
           p_eskf_ukf, d_eskf_ukf, p_eskf_ukf < 0.001 ? "***" : p_eskf_ukf < 0.01 ? "**" : p_eskf_ukf < 0.05 ? "*" : "n.s.");
    printf("  ESKF vs Hybrid: p = %.6f  (Cohen's d = %.2f)  %s\n",
           p_eskf_hybrid, d_eskf_hybrid, p_eskf_hybrid < 0.001 ? "***" : p_eskf_hybrid < 0.01 ? "**" : p_eskf_hybrid < 0.05 ? "*" : "n.s.");
    printf("  UKF vs Hybrid:  p = %.6f  (Cohen's d = %.2f)  %s\n",
           p_ukf_hybrid, d_ukf_hybrid, p_ukf_hybrid < 0.001 ? "***" : p_ukf_hybrid < 0.01 ? "**" : p_ukf_hybrid < 0.05 ? "*" : "n.s.");
    
    printf("\nOrientation RMSE Statistical Comparison:\n");
    p_eskf_ukf = paired_t_test(eskf.ori_rmse_all, ukf.ori_rmse_all);
    p_eskf_hybrid = paired_t_test(eskf.ori_rmse_all, hybrid.ori_rmse_all);
    p_ukf_hybrid = paired_t_test(ukf.ori_rmse_all, hybrid.ori_rmse_all);
    
    d_eskf_ukf = cohens_d(eskf.ori_rmse_all, ukf.ori_rmse_all);
    d_eskf_hybrid = cohens_d(eskf.ori_rmse_all, hybrid.ori_rmse_all);
    d_ukf_hybrid = cohens_d(ukf.ori_rmse_all, hybrid.ori_rmse_all);
    
    printf("  ESKF vs UKF:   p = %.6f  (Cohen's d = %.2f)  %s\n",
           p_eskf_ukf, d_eskf_ukf, p_eskf_ukf < 0.001 ? "***" : p_eskf_ukf < 0.01 ? "**" : p_eskf_ukf < 0.05 ? "*" : "n.s.");
    printf("  ESKF vs Hybrid: p = %.6f  (Cohen's d = %.2f)  %s\n",
           p_eskf_hybrid, d_eskf_hybrid, p_eskf_hybrid < 0.001 ? "***" : p_eskf_hybrid < 0.01 ? "**" : p_eskf_hybrid < 0.05 ? "*" : "n.s.");
    printf("  UKF vs Hybrid:  p = %.6f  (Cohen's d = %.2f)  %s\n",
           p_ukf_hybrid, d_ukf_hybrid, p_ukf_hybrid < 0.001 ? "***" : p_ukf_hybrid < 0.01 ? "**" : p_ukf_hybrid < 0.05 ? "*" : "n.s.");
    
    printf("\n");
    printf("Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. not significant\n");
}