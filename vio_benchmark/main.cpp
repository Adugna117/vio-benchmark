// =============================================================================
// main.cpp  –  VIO Filter Benchmark
//              MSYS2 UCRT64 / GCC C++17
//
// Build:
//   g++ -O3 -std=c++17 -o vio_benchmark main.cpp
//
// Run (EuRoC data):
//   ./vio_benchmark <imu_csv> <gt_csv> [sequence_name]
//
// Run (synthetic demo, no data files needed):
//   ./vio_benchmark --synthetic
// =============================================================================
#include <cstdio>
#include <cstdlib>
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

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic dataset generator (for self-contained testing)
// Simulates a figure-8 trajectory at 200 Hz IMU / 10 Hz camera
// ─────────────────────────────────────────────────────────────────────────────
inline AlignedData make_synthetic(int imu_hz=200, double duration_s=60.0){
    AlignedData ad;
    int N=(int)(imu_hz*duration_s);
    double dt=1.0/imu_hz;

    std::mt19937_64 rng(123);
    std::normal_distribution<double> nd(0,1);

    // True trajectory: horizontal figure-8
    auto traj=[&](double t, Vec3& p, Vec3& v, Vec3& euler)->void{
        double omega=2*M_PI/20.0;  // 20-second loop
        double r=2.0;
        // Lissajous
        p[0]=r*std::sin(omega*t);
        p[1]=r*std::sin(2*omega*t)/2;
        p[2]=0.5*std::sin(omega*t/3)+1.5;  // gentle altitude change
        double c=std::cos(omega*t), s=std::sin(omega*t);
        v[0]=r*omega*c;
        v[1]=r*omega*std::cos(2*omega*t);
        v[2]=0.5*(omega/3)*std::cos(omega*t/3);
        euler[0]=0.1*std::sin(omega*t);    // roll
        euler[1]=0.05*std::sin(2*omega*t); // pitch
        euler[2]=omega*t;                   // yaw
    };

    Vec3 p0,v0,e0;
    traj(0,p0,v0,e0);

    ad.imu_t.resize(N);
    ad.gyro.resize(N);
    ad.accel.resize(N);
    ad.gt_pos.resize(N);
    ad.gt_quat.resize(N);

    double sigma_gyro=0.005, sigma_accel=0.05;

    for(int i=0;i<N;i++){
        double t=i*dt;
        ad.imu_t[i]=t;

        Vec3 p,v,euler;
        traj(t,p,v,euler);
        Vec3 pn,vn,en;
        traj(t+dt,pn,vn,en);

        // true angular rate (finite difference on euler → approximate omega)
        Vec3 omega_true={(en[0]-euler[0])/dt,(en[1]-euler[1])/dt,(en[2]-euler[2])/dt};

        // true accel in body frame: R'*(a_world - g)
        // a_world = (vn-v)/dt
        Vec3 a_world={(vn[0]-v[0])/dt,(vn[1]-v[1])/dt,(vn[2]-v[2])/dt};
        a_world[2]+=9.81;  // add gravity (remove from measurement perspective)

        // Build rotation from euler (ZYX)
        double cr=std::cos(euler[0]),sr=std::sin(euler[0]);
        double cp=std::cos(euler[1]),sp=std::sin(euler[1]);
        double cy=std::cos(euler[2]),sy=std::sin(euler[2]);
        Mat3 R={{{cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr},
                 {sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr},
                 {-sp,   cp*sr,          cp*cr}}};
        Vec3 a_body=transpose(R)*a_world;

        // Add noise
        ad.gyro[i]={omega_true[0]+sigma_gyro*nd(rng),
                    omega_true[1]+sigma_gyro*nd(rng),
                    omega_true[2]+sigma_gyro*nd(rng)};
        ad.accel[i]={a_body[0]+sigma_accel*nd(rng),
                     a_body[1]+sigma_accel*nd(rng),
                     a_body[2]+sigma_accel*nd(rng)};

        ad.gt_pos[i]=p;
        ad.gt_quat[i]=rotm2quat(R);
    }

    // Camera measurements every cam_stride IMU steps
    int cam_stride=imu_hz/10; // 10 Hz camera
    for(int i=0;i<N;i+=cam_stride){
        ad.cam_t.push_back(ad.imu_t[i]);
        ad.cam_pos.push_back(ad.gt_pos[i]);
        ad.cam_quat.push_back(ad.gt_quat[i]);
    }

    return ad;
}

// ─────────────────────────────────────────────────────────────────────────────
// Print comparison table
// ─────────────────────────────────────────────────────────────────────────────
inline void print_table(const std::vector<FilterResult>& results){
    printf("\n");
    printf("========================================\n");
    printf("FINAL COMPARISON RESULTS\n");
    printf("========================================\n");
    printf("| %-17s | %-12s | %-12s | %-10s |\n",
           "Filter Type","Pos RMSE (m)","Ori RMSE (°)","Time (ms)");
    printf("|-------------------|--------------|--------------|------------|\n");
    for(auto& r:results)
        printf("| %-17s | %12.3f | %12.2f | %10.4f |\n",
               r.name.c_str(), r.pos_rmse, r.ori_rmse_deg, r.avg_time_ms);
    printf("========================================\n");

    // Speedup relative to Full UKF
    double ukf_t=-1;
    for(auto& r:results) if(r.name=="Full_UKF") ukf_t=r.avg_time_ms;
    if(ukf_t>0){
        for(auto& r:results) if(r.name!="Full_UKF" && r.avg_time_ms>0){
            double speedup=ukf_t/r.avg_time_ms;
            printf("  %-17s is %.1fx faster than Full UKF (%.1f%% reduction)\n",
                   r.name.c_str(),speedup,(1-1.0/speedup)*100);
        }
    }
    printf("\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Save results to CSV
// ─────────────────────────────────────────────────────────────────────────────
inline void save_csv(const FilterResult& r, const AlignedData& ad,
                     const std::string& fname){
    std::ofstream f(fname);
    f<<"time,est_px,est_py,est_pz,gt_px,gt_py,gt_pz,"
      <<"est_qx,est_qy,est_qz,est_qw,timing_ms\n";
    int N=(int)r.pos.size();
    for(int i=0;i<N;i++){
        f<<ad.imu_t[i]<<','
         <<r.pos[i][0]<<','<<r.pos[i][1]<<','<<r.pos[i][2]<<','
         <<ad.gt_pos[i][0]<<','<<ad.gt_pos[i][1]<<','<<ad.gt_pos[i][2]<<','
         <<r.quat[i][0]<<','<<r.quat[i][1]<<','<<r.quat[i][2]<<','<<r.quat[i][3]<<','
         <<r.timing_ms[i]<<'\n';
    }
    printf("  Saved: %s\n",fname.c_str());
}

inline void save_summary_csv(const std::vector<FilterResult>& results,
                              const std::string& fname){
    std::ofstream f(fname);
    f<<"filter,pos_rmse_m,ori_rmse_deg,avg_time_ms\n";
    for(auto& r:results)
        f<<r.name<<','<<r.pos_rmse<<','<<r.ori_rmse_deg<<','<<r.avg_time_ms<<'\n';
    printf("  Summary saved: %s\n",fname.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]){
    printf("========================================\n");
    printf("VIO Filter Benchmark\n");
    printf("ESKF vs Full UKF vs Hybrid ESKF/UKF\n");
    printf("C++17 | MSYS2 UCRT64\n");
    printf("========================================\n\n");

    AlignedData ad;
    std::string seq_name="synthetic";

    if(argc>=2 && std::string(argv[1])=="--synthetic"){
        printf("[Mode] Synthetic figure-8 trajectory (200 Hz IMU, 60 s)\n\n");
        ad=make_synthetic(200,60.0);
        seq_name="synthetic";
    } else if(argc>=3){
        std::string imu_csv=argv[1];
        std::string gt_csv=argv[2];
        if(argc>=4) seq_name=argv[3];
        printf("[Mode] EuRoC dataset: %s\n",seq_name.c_str());
        printf("  IMU: %s\n  GT : %s\n\n",imu_csv.c_str(),gt_csv.c_str());
        try {
            Dataset ds=load_euroc(imu_csv,gt_csv);
            printf("  Loaded %zu IMU samples, %zu GT samples\n",
                   ds.imu.size(),ds.gt.size());
            ad=align_dataset(ds);
        } catch(std::exception& e){
            fprintf(stderr,"Error loading data: %s\n",e.what());
            return 1;
        }
    } else {
        printf("Usage:\n");
        printf("  %s --synthetic\n",argv[0]);
        printf("  %s <imu.csv> <groundtruth.csv> [sequence_name]\n\n",argv[0]);
        printf("EuRoC CSV format expected.\n");
        printf("Running synthetic demo...\n\n");
        ad=make_synthetic(200,60.0);
    }

    printf("Dataset summary:\n");
    printf("  IMU samples  : %zu\n",ad.imu_t.size());
    printf("  Camera frames: %zu\n",ad.cam_t.size());
    printf("  Duration     : %.2f s\n",ad.imu_t.back()-ad.imu_t.front());
    printf("  IMU rate     : ~%.0f Hz\n",
           (double)(ad.imu_t.size()-1)/(ad.imu_t.back()-ad.imu_t.front()));
    printf("\n");

    // ── Run all three filters ─────────────────────────────────────────────────
    printf("----------------------------------------\n");
    auto t_all=std::chrono::high_resolution_clock::now();

    FilterResult r_eskf   = run_eskf(ad);
    FilterResult r_ukf    = run_full_ukf(ad);
    FilterResult r_hybrid = run_hybrid_filter(ad);

    double total_s=std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now()-t_all).count();
    printf("\nTotal benchmark time: %.2f s\n",total_s);

    // ── Print summary table ───────────────────────────────────────────────────
    std::vector<FilterResult> all={r_eskf,r_ukf,r_hybrid};
    print_table(all);

    // ── Save CSVs ─────────────────────────────────────────────────────────────
    save_csv(r_eskf,   ad, seq_name+"_eskf.csv");
    save_csv(r_ukf,    ad, seq_name+"_full_ukf.csv");
    save_csv(r_hybrid, ad, seq_name+"_hybrid.csv");
    save_summary_csv(all, seq_name+"_summary.csv");

    printf("\nDone.\n");
    return 0;
}
