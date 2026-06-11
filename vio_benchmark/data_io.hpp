#pragma once
// =============================================================================
// data_io.hpp  –  EuRoC CSV loader + shared data structures
// =============================================================================
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "math_utils.hpp"

// ── Raw sensor data ───────────────────────────────────────────────────────────
struct ImuSample {
    double t;
    Vec3   gyro;
    Vec3   accel;
};

struct GtSample {
    double t;
    Vec3   pos;
    Vec4   quat;   // [qx qy qz qw]
};

// ── Loaded dataset ────────────────────────────────────────────────────────────
struct Dataset {
    std::vector<ImuSample> imu;
    std::vector<GtSample>  gt;
};

// ── EuRoC CSV parsers ─────────────────────────────────────────────────────────
// IMU CSV format: timestamp[ns], omega_x, omega_y, omega_z, a_x, a_y, a_z
inline Dataset load_euroc(const std::string& imu_csv, const std::string& gt_csv){
    Dataset ds;

    // --- IMU ---
    {
        std::ifstream f(imu_csv);
        if(!f) throw std::runtime_error("Cannot open IMU file: "+imu_csv);
        std::string line;
        std::getline(f,line); // header
        while(std::getline(f,line)){
            if(line.empty()) continue;
            std::replace(line.begin(),line.end(),',',' ');
            std::istringstream ss(line);
            long long ts; double gx,gy,gz,ax,ay,az;
            ss>>ts>>gx>>gy>>gz>>ax>>ay>>az;
            ds.imu.push_back({ts*1e-9,{gx,gy,gz},{ax,ay,az}});
        }
    }

    // --- Ground truth ---
    // Format: timestamp[ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z, ...
    {
        std::ifstream f(gt_csv);
        if(!f) throw std::runtime_error("Cannot open GT file: "+gt_csv);
        std::string line;
        std::getline(f,line); // header
        while(std::getline(f,line)){
            if(line.empty()) continue;
            std::replace(line.begin(),line.end(),',',' ');
            std::istringstream ss(line);
            long long ts;
            double px,py,pz,qw,qx,qy,qz;
            ss>>ts>>px>>py>>pz>>qw>>qx>>qy>>qz;
            // EuRoC GT quaternion is [qw qx qy qz] → convert to [qx qy qz qw]
            ds.gt.push_back({ts*1e-9,{px,py,pz},{qx,qy,qz,qw}});
        }
    }

    // Sort by time (should already be sorted)
    std::sort(ds.imu.begin(),ds.imu.end(),[](auto&a,auto&b){return a.t<b.t;});
    std::sort(ds.gt.begin(), ds.gt.end(), [](auto&a,auto&b){return a.t<b.t;});

    return ds;
}

// ── Linear interpolation helpers ─────────────────────────────────────────────
inline Vec3 interp_vec3(const std::vector<double>& ts,
                        const std::vector<Vec3>& vs,
                        double t){
    if(t<=ts.front()) return vs.front();
    if(t>=ts.back())  return vs.back();
    auto it=std::lower_bound(ts.begin(),ts.end(),t);
    int hi=(int)(it-ts.begin()), lo=hi-1;
    double alpha=(t-ts[lo])/(ts[hi]-ts[lo]);
    return (1-alpha)*vs[lo] + alpha*vs[hi];
}
inline Vec4 interp_quat(const std::vector<double>& ts,
                        const std::vector<Vec4>& qs,
                        double t){
    if(t<=ts.front()) return qs.front();
    if(t>=ts.back())  return qs.back();
    auto it=std::lower_bound(ts.begin(),ts.end(),t);
    int hi=(int)(it-ts.begin()), lo=hi-1;
    double alpha=(t-ts[lo])/(ts[hi]-ts[lo]);
    // SLERP
    Vec4 q0=qs[lo], q1=qs[hi];
    double d=q0[0]*q1[0]+q0[1]*q1[1]+q0[2]*q1[2]+q0[3]*q1[3];
    if(d<0){ q1={-q1[0],-q1[1],-q1[2],-q1[3]}; d=-d; }
    if(d>0.9995){
        // linear approx
        Vec4 q={q0[0]+alpha*(q1[0]-q0[0]),
                q0[1]+alpha*(q1[1]-q0[1]),
                q0[2]+alpha*(q1[2]-q0[2]),
                q0[3]+alpha*(q1[3]-q0[3])};
        return qnorm(q);
    }
    double theta0=std::acos(d);
    double theta=theta0*alpha;
    double s0=std::cos(theta)-d*std::sin(theta)/std::sin(theta0);
    double s1=std::sin(theta)/std::sin(theta0);
    return qnorm({s0*q0[0]+s1*q1[0], s0*q0[1]+s1*q1[1],
                  s0*q0[2]+s1*q1[2], s0*q0[3]+s1*q1[3]});
}

// ── Aligned dataset (IMU-rate, GT interpolated to IMU times) ─────────────────
struct AlignedData {
    std::vector<double> imu_t;
    std::vector<Vec3>   gyro, accel;
    std::vector<Vec3>   gt_pos;
    std::vector<Vec4>   gt_quat;
    // Camera measurement times (every 20 IMU steps)
    std::vector<double> cam_t;
    std::vector<Vec3>   cam_pos;
    std::vector<Vec4>   cam_quat;
};

inline AlignedData align_dataset(const Dataset& ds, int cam_stride=20){
    AlignedData ad;
    double t0=std::max(ds.imu.front().t, ds.gt.front().t);
    double t1=std::min(ds.imu.back().t,  ds.gt.back().t);

    // Build GT lookup vectors
    std::vector<double> gt_ts;
    std::vector<Vec3>   gt_ps;
    std::vector<Vec4>   gt_qs;
    for(auto& s:ds.gt) if(s.t>=t0 && s.t<=t1){
        gt_ts.push_back(s.t);
        gt_ps.push_back(s.pos);
        gt_qs.push_back(s.quat);
    }

    // Collect valid IMU and interpolate GT
    int idx=0;
    for(auto& s:ds.imu){
        if(s.t<t0 || s.t>t1) continue;
        ad.imu_t.push_back(s.t);
        ad.gyro.push_back(s.gyro);
        ad.accel.push_back(s.accel);
        ad.gt_pos.push_back(interp_vec3(gt_ts,gt_ps,s.t));
        ad.gt_quat.push_back(interp_quat(gt_ts,gt_qs,s.t));

        // Camera pseudo-measurements
        if(idx % cam_stride == 0){
            ad.cam_t.push_back(s.t);
            ad.cam_pos.push_back(ad.gt_pos.back());
            ad.cam_quat.push_back(ad.gt_quat.back());
        }
        ++idx;
    }
    return ad;
}

// ── Filter result storage ────────────────────────────────────────────────────
struct FilterResult {
    std::string name;
    std::vector<Vec3> pos;
    std::vector<Vec4> quat;
    std::vector<double> timing_ms; // per-step wall time
    double pos_rmse{};
    double ori_rmse_deg{};
    double avg_time_ms{};
};

// ── RMSE computation ─────────────────────────────────────────────────────────
inline void compute_rmse(FilterResult& res, const AlignedData& ad){
    int N=(int)res.pos.size();
    double sum_pos=0, sum_ori=0;
    for(int i=0;i<N;i++){
        Vec3 dp=res.pos[i]-ad.gt_pos[i];
        sum_pos+=dot(dp,dp);
        Mat3 Re=quat2rotm(res.quat[i]);
        Mat3 Rg=quat2rotm(ad.gt_quat[i]);
        double ang=rot_angle_deg(Re,Rg);
        sum_ori+=ang*ang;
    }
    res.pos_rmse=std::sqrt(sum_pos/N);
    res.ori_rmse_deg=std::sqrt(sum_ori/N);

    // Average timing (exclude outliers >10ms)
    double st=0; int cnt=0;
    for(double t:res.timing_ms) if(t>0 && t<10){st+=t;cnt++;}
    res.avg_time_ms=(cnt>0)?(st/cnt):0.0;
}
// ← NO EXTRA CODE AFTER THIS CLOSING BRACE