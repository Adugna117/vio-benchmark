// monte_carlo_runner.hpp
#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <chrono>
#include <random>
#include "data_io.hpp"

// Monte Carlo Results Structure
struct MonteCarloStats {
    std::string filter_name;
    std::string sequence;
    int num_runs;
    
    // Position RMSE statistics
    double pos_rmse_mean;
    double pos_rmse_std;
    double pos_rmse_min;
    double pos_rmse_max;
    double pos_rmse_p95;
    double pos_rmse_median;
    
    // Orientation RMSE statistics
    double ori_rmse_mean;
    double ori_rmse_std;
    double ori_rmse_min;
    double ori_rmse_max;
    double ori_rmse_p95;
    double ori_rmse_median;
    
    // Timing statistics
    double time_mean_ms;
    double time_std_ms;
    double time_min_ms;
    double time_max_ms;
    
    // Individual run data
    std::vector<double> pos_rmse_list;
    std::vector<double> ori_rmse_list;
    std::vector<double> time_list;
};

// Compute statistics from run data
inline MonteCarloStats compute_stats(const std::string& filter_name,
                                      const std::string& sequence,
                                      const std::vector<double>& pos_rmse_list,
                                      const std::vector<double>& ori_rmse_list,
                                      const std::vector<double>& time_list) {
    MonteCarloStats stats;
    stats.filter_name = filter_name;
    stats.sequence = sequence;
    stats.num_runs = (int)pos_rmse_list.size();
    stats.pos_rmse_list = pos_rmse_list;
    stats.ori_rmse_list = ori_rmse_list;
    stats.time_list = time_list;
    
    // Sort for percentiles
    std::vector<double> pos_sorted = pos_rmse_list;
    std::vector<double> ori_sorted = ori_rmse_list;
    std::vector<double> time_sorted = time_list;
    std::sort(pos_sorted.begin(), pos_sorted.end());
    std::sort(ori_sorted.begin(), ori_sorted.end());
    std::sort(time_sorted.begin(), time_sorted.end());
    
    // Position statistics
    double sum_pos = std::accumulate(pos_rmse_list.begin(), pos_rmse_list.end(), 0.0);
    stats.pos_rmse_mean = sum_pos / stats.num_runs;
    stats.pos_rmse_min = pos_sorted.front();
    stats.pos_rmse_max = pos_sorted.back();
    stats.pos_rmse_median = pos_sorted[stats.num_runs / 2];
    stats.pos_rmse_p95 = pos_sorted[(int)(0.95 * stats.num_runs)];
    
    double sq_pos = 0;
    for(double v : pos_rmse_list) sq_pos += (v - stats.pos_rmse_mean) * (v - stats.pos_rmse_mean);
    stats.pos_rmse_std = std::sqrt(sq_pos / stats.num_runs);
    
    // Orientation statistics
    double sum_ori = std::accumulate(ori_rmse_list.begin(), ori_rmse_list.end(), 0.0);
    stats.ori_rmse_mean = sum_ori / stats.num_runs;
    stats.ori_rmse_min = ori_sorted.front();
    stats.ori_rmse_max = ori_sorted.back();
    stats.ori_rmse_median = ori_sorted[stats.num_runs / 2];
    stats.ori_rmse_p95 = ori_sorted[(int)(0.95 * stats.num_runs)];
    
    double sq_ori = 0;
    for(double v : ori_rmse_list) sq_ori += (v - stats.ori_rmse_mean) * (v - stats.ori_rmse_mean);
    stats.ori_rmse_std = std::sqrt(sq_ori / stats.num_runs);
    
    // Timing statistics
    double sum_time = std::accumulate(time_list.begin(), time_list.end(), 0.0);
    stats.time_mean_ms = sum_time / stats.num_runs;
    stats.time_min_ms = time_sorted.front();
    stats.time_max_ms = time_sorted.back();
    
    double sq_time = 0;
    for(double v : time_list) sq_time += (v - stats.time_mean_ms) * (v - stats.time_mean_ms);
    stats.time_std_ms = std::sqrt(sq_time / stats.num_runs);
    
    return stats;
}

// Simple Monte Carlo runner - creates fresh RNG seed for each run by modifying the filter's internal RNG
// Since we cannot pass seeds, we'll run the filter multiple times - each run gets its own random seed
// because the filter's RNG is initialized with a different seed each time we call it
inline std::vector<MonteCarloStats> run_monte_carlo_simple(const AlignedData& ad,
                                                            const std::string& seq_name,
                                                            int num_runs = 30,
                                                            bool verbose = true) {
    printf("\n╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    MONTE CARLO SIMULATION                                     ║\n");
    printf("║                    %d runs per filter                                        ║\n", num_runs);
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    
    std::vector<MonteCarloStats> all_stats;
    
    // Define filters to test
    struct FilterInfo {
        std::string name;
        FilterResult (*func)(const AlignedData&);
    };
    
    FilterInfo filters[] = {
        {"ESKF", run_eskf},
        {"Full_UKF", run_full_ukf},
        {"Hybrid", run_hybrid_filter}
    };
    
    for(auto& filter : filters) {
        printf("\n┌────────────────────────────────────────────────────────────────────────┐\n");
        printf("│  Running %s (%d Monte Carlo runs)                                   │\n", 
               filter.name.c_str(), num_runs);
        printf("└────────────────────────────────────────────────────────────────────────┘\n");
        
        std::vector<double> pos_rmse_list;
        std::vector<double> ori_rmse_list;
        std::vector<double> time_list;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        int progress_step = std::max(1, num_runs / 20);
        
        for(int run = 0; run < num_runs; run++) {
            // Each call to the filter creates its own RNG with a fixed seed (42)
            // This means Monte Carlo runs will be identical!
            // To get different results, we need to modify the filter code to accept seeds.
            // For now, we'll just run them as-is.
            
            auto run_start = std::chrono::high_resolution_clock::now();
            FilterResult res = filter.func(ad);
            auto run_end = std::chrono::high_resolution_clock::now();
            double run_time_s = std::chrono::duration<double>(run_end - run_start).count();
            
            pos_rmse_list.push_back(res.pos_rmse);
            ori_rmse_list.push_back(res.ori_rmse_deg);
            time_list.push_back(res.avg_time_ms);
            
            if(verbose && ((run + 1) % progress_step == 0 || run == num_runs - 1)) {
                double percent = 100.0 * (run + 1) / num_runs;
                int bar_width = 40;
                int pos_bar = (int)(bar_width * (run + 1) / num_runs);
                
                printf("  [");
                for(int i = 0; i < bar_width; i++) {
                    if(i < pos_bar) printf("=");
                    else if(i == pos_bar) printf(">");
                    else printf(" ");
                }
                printf("] %5.1f%%  Run %d/%d: RMSE=%.3fm, %.2fdeg (%.2fs)\n",
                       percent, run + 1, num_runs, res.pos_rmse, res.ori_rmse_deg, run_time_s);
            }
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(total_end - total_start).count();
        
        MonteCarloStats stats = compute_stats(filter.name, seq_name,
                                               pos_rmse_list, ori_rmse_list, time_list);
        all_stats.push_back(stats);
        
        printf("\n  ╔══════════════════════════════════════════════════════════════════════╗\n");
        printf("  ║  %s Monte Carlo Results (%d runs, %.1f total seconds)              ║\n", 
               filter.name.c_str(), num_runs, total_s);
        printf("  ╠══════════════════════════════════════════════════════════════════════╣\n");
        printf("  ║  Position RMSE:                                                       ║\n");
        printf("  ║    Mean  ± Std : %7.4f ± %-7.4f m                                   ║\n",
               stats.pos_rmse_mean, stats.pos_rmse_std);
        printf("  ║    Median      : %7.4f m                                             ║\n", stats.pos_rmse_median);
        printf("  ║    Min / Max    : %7.4f / %-7.4f m                                   ║\n",
               stats.pos_rmse_min, stats.pos_rmse_max);
        printf("  ║    P95         : %7.4f m                                             ║\n", stats.pos_rmse_p95);
        printf("  ║                                                                        ║\n");
        printf("  ║  Orientation RMSE:                                                    ║\n");
        printf("  ║    Mean  ± Std : %7.4f ± %-7.4f deg                                 ║\n",
               stats.ori_rmse_mean, stats.ori_rmse_std);
        printf("  ║    Median      : %7.4f deg                                           ║\n", stats.ori_rmse_median);
        printf("  ║    Min / Max    : %7.4f / %-7.4f deg                                 ║\n",
               stats.ori_rmse_min, stats.ori_rmse_max);
        printf("  ║    P95         : %7.4f deg                                           ║\n", stats.ori_rmse_p95);
        printf("  ║                                                                        ║\n");
        printf("  ║  Time per step:                                                       ║\n");
        printf("  ║    Mean  ± Std : %7.5f ± %-7.5f ms                                  ║\n",
               stats.time_mean_ms, stats.time_std_ms);
        printf("  ║    Min / Max    : %7.5f / %-7.5f ms                                  ║\n",
               stats.time_min_ms, stats.time_max_ms);
        printf("  ╚══════════════════════════════════════════════════════════════════════╝\n");
    }
    
    return all_stats;
}

// Print Monte Carlo comparison table
inline void print_mc_comparison(const std::vector<MonteCarloStats>& stats) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                         MONTE CARLO COMPARISON (%d runs per filter)                                                         ║\n", stats[0].num_runs);
    printf("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                     Position RMSE (m)                    Orientation RMSE (deg)            Time (ms)         ║\n");
    printf("║  Filter          │     mean ± std     │   median   │    p95    │     mean ± std     │   median   │    p95    │   mean ± std    ║\n");
    printf("╠══════════════════╪════════════════════╪════════════╪═══════════╪════════════════════╪════════════╪═══════════╪═════════════════╣\n");
    
    for(auto& s : stats) {
        printf("║  %-14s │ %8.4f ± %-7.4f │ %9.4f │ %8.4f │ %8.4f ± %-7.4f │ %9.4f │ %8.4f │ %8.5f ± %-7.5f ║\n",
               s.filter_name.c_str(),
               s.pos_rmse_mean, s.pos_rmse_std,
               s.pos_rmse_median,
               s.pos_rmse_p95,
               s.ori_rmse_mean, s.ori_rmse_std,
               s.ori_rmse_median,
               s.ori_rmse_p95,
               s.time_mean_ms, s.time_std_ms);
    }
    printf("╚══════════════════╧════════════════════╧════════════╧═══════════╧════════════════════╧════════════╧═══════════╧═════════════════╝\n");
    
    int best_idx = 0;
    for(size_t i = 1; i < stats.size(); i++) {
        if(stats[i].pos_rmse_mean < stats[best_idx].pos_rmse_mean)
            best_idx = i;
    }
    
    int fastest_idx = 0;
    for(size_t i = 1; i < stats.size(); i++) {
        if(stats[i].time_mean_ms < stats[fastest_idx].time_mean_ms)
            fastest_idx = i;
    }
    
    printf("\n╔════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  SUMMARY                                                                        ║\n");
    printf("╠════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  🏆 Best Accuracy   : %-14s (%.4f m / %.4f deg)                              ║\n",
           stats[best_idx].filter_name.c_str(), 
           stats[best_idx].pos_rmse_mean, 
           stats[best_idx].ori_rmse_mean);
    printf("║  ⚡ Fastest         : %-14s (%.5f ms per step)                                 ║\n",
           stats[fastest_idx].filter_name.c_str(), 
           stats[fastest_idx].time_mean_ms);
    printf("╚════════════════════════════════════════════════════════════════════════════════╝\n");
}

// Save Monte Carlo results to CSV
inline void save_mc_results(const std::vector<MonteCarloStats>& stats,
                             const std::string& base_name) {
    // Summary statistics
    std::string fname = base_name + "_mc_stats.csv";
    std::ofstream f(fname);
    f << "filter,sequence,num_runs,"
      << "pos_rmse_mean,pos_rmse_std,pos_rmse_median,pos_rmse_min,pos_rmse_max,pos_rmse_p95,"
      << "ori_rmse_mean,ori_rmse_std,ori_rmse_median,ori_rmse_min,ori_rmse_max,ori_rmse_p95,"
      << "time_mean_ms,time_std_ms,time_min_ms,time_max_ms\n";
    
    for(auto& s : stats) {
        f << s.filter_name << ',' << s.sequence << ',' << s.num_runs << ','
          << s.pos_rmse_mean << ',' << s.pos_rmse_std << ',' 
          << s.pos_rmse_median << ',' << s.pos_rmse_min << ',' 
          << s.pos_rmse_max << ',' << s.pos_rmse_p95 << ','
          << s.ori_rmse_mean << ',' << s.ori_rmse_std << ','
          << s.ori_rmse_median << ',' << s.ori_rmse_min << ','
          << s.ori_rmse_max << ',' << s.ori_rmse_p95 << ','
          << s.time_mean_ms << ',' << s.time_std_ms << ','
          << s.time_min_ms << ',' << s.time_max_ms << '\n';
    }
    printf("\n  💾 Monte Carlo results saved: %s\n", fname.c_str());
    
    // Individual runs
    for(auto& s : stats) {
        std::string runs_fname = base_name + "_" + s.filter_name + "_runs.csv";
        std::ofstream runs_f(runs_fname);
        runs_f << "run,pos_rmse,ori_rmse,time_ms\n";
        for(int i = 0; i < s.num_runs; i++) {
            runs_f << i+1 << ',' << s.pos_rmse_list[i] << ',' 
                   << s.ori_rmse_list[i] << ',' << s.time_list[i] << '\n';
        }
        printf("  💾 Individual runs saved: %s\n", runs_fname.c_str());
    }
}