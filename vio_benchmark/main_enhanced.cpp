// =============================================================================
// main.cpp  –  VIO Filter Benchmark  (Enhanced with Full Metrics)
//              MSYS2 UCRT64 / GCC C++17
//
// Build:
//   g++ -O3 -std=c++17 -march=native -ffast-math -funroll-loops \
//       -D_USE_MATH_DEFINES -o vio_benchmark.exe main.cpp -lpsapi
//
// Run:
//   ./vio_benchmark.exe --synthetic
//   ./vio_benchmark.exe <imu.csv> <gt.csv> <seq_name>
// =============================================================================
#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>

#include "math_utils.hpp"
#include "data_io.hpp"
#include "eskf.hpp"
#include "full_ukf.hpp"
#include "hybrid_filter.hpp"
#include "benchmark_metrics.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic dataset (figure-8, 200 Hz IMU, 10 Hz camera)
// ─────────────────────────────────────────────────────────────────────────────
inline AlignedData make_synthetic(int imu_hz=200, double duration_s=60.0){
    AlignedData ad;
    int N=(int)(imu_hz*duration_s);
    double dt=1.0/imu_hz;
    std::mt19937_64 rng(123);
    std::normal_distribution<double> nd(0,1);

    auto traj=[&](double t, Vec3& p, Vec3& v, Vec3& euler){
        double omega=2*M_PI/20.0;
        double r=2.0;
        p[0]=r*std::sin(omega*t);
        p[1]=r*std::sin(2*omega*t)/2;
        p[2]=0.5*std::sin(omega*t/3)+1.5;
        v[0]=r*omega*std::cos(omega*t);
        v[1]=r*omega*std::cos(2*omega*t);
        v[2]=0.5*(omega/3)*std::cos(omega*t/3);
        euler[0]=0.1*std::sin(omega*t);
        euler[1]=0.05*std::sin(2*omega*t);
        euler[2]=omega*t;
    };

    ad.imu_t.resize(N); ad.gyro.resize(N); ad.accel.resize(N);
    ad.gt_pos.resize(N); ad.gt_quat.resize(N);

    double sg=0.005, sa=0.05;
    for(int i=0;i<N;i++){
        double t=i*dt;
        ad.imu_t[i]=t;
        Vec3 p,v,e,pn,vn,en;
        traj(t,p,v,e); traj(t+dt,pn,vn,en);
        Vec3 omega_true={(en[0]-e[0])/dt,(en[1]-e[1])/dt,(en[2]-e[2])/dt};
        Vec3 a_world={(vn[0]-v[0])/dt,(vn[1]-v[1])/dt,(vn[2]-v[2])/dt};
        a_world[2]+=9.81;
        double cr=std::cos(e[0]),sr=std::sin(e[0]);
        double cp=std::cos(e[1]),sp=std::sin(e[1]);
        double cy=std::cos(e[2]),sy=std::sin(e[2]);
        Mat3 R={{{cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr},
                 {sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr},
                 {-sp,  cp*sr,         cp*cr}}};
        Vec3 a_body=transpose(R)*a_world;
        ad.gyro[i]={omega_true[0]+sg*nd(rng),omega_true[1]+sg*nd(rng),omega_true[2]+sg*nd(rng)};
        ad.accel[i]={a_body[0]+sa*nd(rng),a_body[1]+sa*nd(rng),a_body[2]+sa*nd(rng)};
        ad.gt_pos[i]=p;
        ad.gt_quat[i]=rotm2quat(R);
    }
    int cam_stride=imu_hz/10;
    for(int i=0;i<N;i+=cam_stride){
        ad.cam_t.push_back(ad.imu_t[i]);
        ad.cam_pos.push_back(ad.gt_pos[i]);
        ad.cam_quat.push_back(ad.gt_quat[i]);
    }
    return ad;
}

// ─────────────────────────────────────────────────────────────────────────────
// Save trajectory CSV
// ─────────────────────────────────────────────────────────────────────────────
inline void save_csv(const FilterResult& r, const AlignedData& ad,
                     const std::string& fname){
    std::ofstream f(fname);
    f<<"time,est_px,est_py,est_pz,gt_px,gt_py,gt_pz,"
      "est_qx,est_qy,est_qz,est_qw,timing_ms\n";
    int N=(int)r.pos.size();
    for(int i=0;i<N;i++)
        f<<ad.imu_t[i]<<','
         <<r.pos[i][0]<<','<<r.pos[i][1]<<','<<r.pos[i][2]<<','
         <<ad.gt_pos[i][0]<<','<<ad.gt_pos[i][1]<<','<<ad.gt_pos[i][2]<<','
         <<r.quat[i][0]<<','<<r.quat[i][1]<<','<<r.quat[i][2]<<','<<r.quat[i][3]<<','
         <<r.timing_ms[i]<<'\n';
    printf("  Saved: %s\n",fname.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Save summary CSV
// ─────────────────────────────────────────────────────────────────────────────
inline void save_summary_csv(const std::vector<FilterResult>& results,
                              const std::string& fname){
    std::ofstream f(fname);
    f<<"filter,pos_rmse_m,ori_rmse_deg,avg_time_ms\n";
    for(auto& r:results)
        f<<r.name<<','<<r.pos_rmse<<','<<r.ori_rmse_deg<<','<<r.avg_time_ms<<'\n';
    printf("  Summary saved: %s\n",fname.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Print final table
// ─────────────────────────────────────────────────────────────────────────────
inline void print_table(const std::vector<FilterResult>& results){
    printf("\n========================================\n");
    printf("FINAL COMPARISON RESULTS\n");
    printf("========================================\n");
    printf("| %-17s | %-12s | %-12s | %-10s |\n",
           "Filter Type","Pos RMSE (m)","Ori RMSE (deg)","Time (ms)");
    printf("|-------------------|--------------|--------------|------------|\n");
    for(auto& r:results)
        printf("| %-17s | %12.3f | %12.2f | %10.4f |\n",
               r.name.c_str(),r.pos_rmse,r.ori_rmse_deg,r.avg_time_ms);
    printf("========================================\n");
    double ukf_t=-1;
    for(auto& r:results) if(r.name=="Full_UKF") ukf_t=r.avg_time_ms;
    if(ukf_t>0) for(auto& r:results) if(r.name!="Full_UKF" && r.avg_time_ms>0)
        printf("  %-17s is %.1fx faster than Full UKF (%.1f%% reduction)\n",
               r.name.c_str(),ukf_t/r.avg_time_ms,(1-1.0/ukf_t*r.avg_time_ms)*100);
    printf("\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]){
    printf("========================================\n");
    printf("VIO Filter Benchmark  (Enhanced Metrics)\n");
    printf("ESKF vs Full UKF vs Hybrid ESKF/UKF\n");
    printf("C++17 | MSYS2 UCRT64\n");
    printf("========================================\n\n");

    AlignedData ad;
    std::string seq_name="synthetic";

    if(argc>=2 && std::string(argv[1])=="--synthetic"){
        printf("[Mode] Synthetic figure-8 (200Hz, 60s)\n\n");
        ad=make_synthetic(200,60.0);
    } else if(argc>=3){
        std::string imu_csv=argv[1], gt_csv=argv[2];
        if(argc>=4) seq_name=argv[3];
        printf("[Mode] EuRoC: %s\n  IMU: %s\n  GT : %s\n\n",
               seq_name.c_str(),imu_csv.c_str(),gt_csv.c_str());
        try{
            Dataset ds=load_euroc(imu_csv,gt_csv);
            printf("  Loaded %zu IMU, %zu GT samples\n",ds.imu.size(),ds.gt.size());
            ad=align_dataset(ds);
        } catch(std::exception& e){
            fprintf(stderr,"Error: %s\n",e.what()); return 1;
        }
    } else {
        printf("[Mode] Synthetic demo (no args given)\n\n");
        ad=make_synthetic(200,60.0);
    }

    printf("Dataset: %zu IMU samples | %zu camera frames | %.2fs\n\n",
           ad.imu_t.size(), ad.cam_t.size(),
           ad.imu_t.back()-ad.imu_t.front());

    // ── Snapshot resources BEFORE each filter run ─────────────────────────────
    printf("----------------------------------------\n");
    auto t_all=std::chrono::high_resolution_clock::now();

    // ── ESKF ──────────────────────────────────────────────────────────────────
    double mem_b=get_rss_mb(), cpu_b=get_cpu_time_ms();
    FilterResult r_eskf=run_eskf(ad);
    BenchmarkMetrics m_eskf=compute_metrics("ESKF",seq_name,
        r_eskf.timing_ms,r_eskf.pos_rmse,r_eskf.ori_rmse_deg,mem_b,cpu_b);
    print_metrics(m_eskf);

    // ── Full UKF ──────────────────────────────────────────────────────────────
    mem_b=get_rss_mb(); cpu_b=get_cpu_time_ms();
    FilterResult r_ukf=run_full_ukf(ad);
    BenchmarkMetrics m_ukf=compute_metrics("Full_UKF",seq_name,
        r_ukf.timing_ms,r_ukf.pos_rmse,r_ukf.ori_rmse_deg,mem_b,cpu_b);
    print_metrics(m_ukf);

    // ── Hybrid ────────────────────────────────────────────────────────────────
    mem_b=get_rss_mb(); cpu_b=get_cpu_time_ms();
    FilterResult r_hybrid=run_hybrid_filter(ad);
    BenchmarkMetrics m_hybrid=compute_metrics("Hybrid",seq_name,
        r_hybrid.timing_ms,r_hybrid.pos_rmse,r_hybrid.ori_rmse_deg,mem_b,cpu_b);
    print_metrics(m_hybrid);

    double total_s=std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now()-t_all).count();
    printf("\nTotal benchmark wall time: %.2fs\n",total_s);

    // ── Comparison tables ─────────────────────────────────────────────────────
    std::vector<FilterResult>    all_r={r_eskf,r_ukf,r_hybrid};
    std::vector<BenchmarkMetrics> all_m={m_eskf,m_ukf,m_hybrid};

    print_table(all_r);
    print_comparison_table(all_m);

    // ── Save outputs ──────────────────────────────────────────────────────────
    save_csv(r_eskf,   ad, seq_name+"_eskf.csv");
    save_csv(r_ukf,    ad, seq_name+"_full_ukf.csv");
    save_csv(r_hybrid, ad, seq_name+"_hybrid.csv");
    save_summary_csv(all_r, seq_name+"_summary.csv");
    save_metrics_csv(all_m,  seq_name+"_metrics.csv");

    printf("\nDone.\n");
    return 0;
}