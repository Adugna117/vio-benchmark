// main_mc.cpp - Monte Carlo Benchmark for EuRoC Dataset
#define _USE_MATH_DEFINES
#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include "math_utils.hpp"
#include "data_io.hpp"
#include "eskf.hpp"
#include "full_ukf.hpp"
#include "hybrid_filter.hpp"
#include "monte_carlo_runner.hpp"

int main(int argc, char* argv[]){
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                VIO Filter Monte Carlo Benchmark                               ║\n");
    printf("║                ESKF vs Full UKF vs Hybrid ESKF/UKF                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    
    if(argc < 4) {
        printf("\n❌ Usage: %s --mc=N <imu.csv> <gt.csv> <sequence_name>\n", argv[0]);
        printf("   Example: %s --mc=30 /path/to/imu.csv /path/to/gt.csv MH01\n\n", argv[0]);
        return 1;
    }
    
    int num_mc_runs = 30;
    std::string imu_csv, gt_csv, seq_name;
    
    for(int i = 1; i < argc; i++) {
        if(strncmp(argv[i], "--mc=", 5) == 0) {
            num_mc_runs = atoi(argv[i] + 5);
        } else if(imu_csv.empty()) {
            imu_csv = argv[i];
        } else if(gt_csv.empty()) {
            gt_csv = argv[i];
        } else {
            seq_name = argv[i];
        }
    }
    
    if(imu_csv.empty() || gt_csv.empty()) {
        printf("\n❌ Error: Please provide IMU and GT CSV files\n");
        return 1;
    }
    
    printf("\n📡 Mode: EuRoC dataset: %s\n", seq_name.c_str());
    printf("   IMU: %s\n", imu_csv.c_str());
    printf("   GT : %s\n", gt_csv.c_str());
    
    AlignedData ad;
    try{
        Dataset ds = load_euroc(imu_csv, gt_csv);
        printf("   Loaded %zu IMU, %zu GT samples\n", ds.imu.size(), ds.gt.size());
        ad = align_dataset(ds);
    } catch(std::exception& e){
        fprintf(stderr, "\n❌ Error: %s\n", e.what()); 
        return 1;
    }
    
    printf("\n📊 Dataset: %zu IMU samples | %zu camera frames | %.2fs\n",
           ad.imu_t.size(), ad.cam_t.size(),
           ad.imu_t.back() - ad.imu_t.front());
    
    auto t_all = std::chrono::high_resolution_clock::now();
    
    std::vector<MonteCarloStats> mc_stats = run_monte_carlo(ad, seq_name, num_mc_runs, true);
    print_mc_comparison(mc_stats);
    save_mc_results(mc_stats, seq_name);
    
    double total_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_all).count();
    printf("\n⏱️  Total Monte Carlo time: %.2f seconds\n", total_s);
    printf("\n✅ Done.\n");
    
    return 0;
}