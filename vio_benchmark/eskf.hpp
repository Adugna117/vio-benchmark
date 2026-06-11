#pragma once
// =============================================================================
// eskf.hpp  –  Error-State Kalman Filter (15 states)
//              Direct port of run_eskf_only.m
// =============================================================================
#include <cstdio>
#include <random>
#include "math_utils.hpp"
#include "data_io.hpp"

// ── Noise-generation helper ───────────────────────────────────────────────────
struct RNG {
    std::mt19937_64 eng{42};
    std::normal_distribution<double> nd{0.0,1.0};
    double randn(){ return nd(eng); }
    Vec3 randn3(){ return {nd(eng),nd(eng),nd(eng)}; }
};

// ── ESKF filter state ─────────────────────────────────────────────────────────
struct ESKFState {
    Vec3  p{}, v{}, b_g{}, b_a{};
    Vec4  q{0,0,0,1};
    Mat3  R = eye3();
    Mat15 P = eye15();
};

inline ESKFState eskf_init(const AlignedData& ad){
    ESKFState s;
    s.p  = ad.gt_pos[0];
    s.q  = qnorm(ad.gt_quat[0]);
    s.R  = quat2rotm(s.q);
    // covariance
    for(int i=0;i<3;i++) s.P[i][i]   = (0.5*M_PI/180)*(0.5*M_PI/180);
    for(int i=3;i<6;i++) s.P[i][i]   = 0.01*0.01;
    for(int i=6;i<9;i++) s.P[i][i]   = 0.05*0.05;
    for(int i=9;i<12;i++) s.P[i][i]  = 0.001*0.001;
    for(int i=12;i<15;i++) s.P[i][i] = 0.01*0.01;
    return s;
}

inline void apply_zupt(ESKFState& s, Vec3 omega_raw, Vec3 acc_raw){
    bool stationary = (norm(omega_raw)<0.01) && (std::abs(norm(acc_raw)-9.81)<0.05);
    if(!stationary) return;
    // H = zeros(3,15),  H[0:3][6:9]=I
    H3 H{};
    H[0][6]=1; H[1][7]=1; H[2][8]=1;
    double sig2=0.01*0.01;
    Mat3 S = mat_HPH(H,s.P);
    for(int i=0;i<3;i++) S[i][i]+=sig2;
    // symmetrize 3x3
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) S[i][j]=(S[i][j]+S[j][i])/2;
    for(int i=0;i<3;i++) S[i][i]+=1e-6;

    PH3 PH=mat_PH(s.P,H);
    PH3 K=mat_Kgain(PH,inv3(S));
    // gain scaling
    for(int i=0;i<15;i++) for(int j=0;j<3;j++) K[i][j]*=0.3;

    Vec3 innov = {-s.v[0],-s.v[1],-s.v[2]};
    Vec15 dx=mat_Kinnov(K,innov);
    s.v[0]+=dx[6]; s.v[1]+=dx[7]; s.v[2]+=dx[8];
    s.P=joseph_update(s.P,K,H,sig2);
}

inline void apply_gravity_align(ESKFState& s, Vec3 acc_raw){
    bool low_acc = std::abs(norm(acc_raw)-9.81)<0.1;
    if(!low_acc) return;
    Vec3 g_meas=normalized(acc_raw);
    Vec3 g_exp =s.R[0][2]<-0.5 || s.R[1][2]<-0.5 || s.R[2][2]<-0.5 ?
                    Vec3{0,0,-1} : Vec3{0,0,-1}; // always downward unit
    // g_expected = R' * [0;0;-1]
    g_exp = {-s.R[0][2], -s.R[1][2], -s.R[2][2]};
    Vec3 innov = g_meas - g_exp;

    H3 H{};
    // H_gravity(1:3,1:3) = skew(g_exp)
    Mat3 sk=skew(g_exp);
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) H[i][j]=sk[i][j];

    double sig2=0.05*0.05;
    Mat3 S=mat_HPH(H,s.P);
    for(int i=0;i<3;i++) S[i][i]+=sig2+1e-6;

    PH3 PH=mat_PH(s.P,H);
    PH3 K=mat_Kgain(PH,inv3(S));
    for(int i=0;i<15;i++) for(int j=0;j<3;j++) K[i][j]*=0.5;

    Vec15 dx=mat_Kinnov(K,innov);
    Vec3 dphi={dx[0],dx[1],dx[2]};
    s.R=s.R*expm_so3(dphi);
    s.q=rotm2quat(s.R);
    s.P=joseph_update(s.P,K,H,sig2);
}

// ── Process noise matrix ─────────────────────────────────────────────────────
inline Mat15 make_Q_eskf(){
    double sg=0.0002,sa=0.002,sbg=0.00001,sba=0.0001;
    double tau_g=100,tau_a=100;
    Mat15 Q;
    for(int i=0;i<3;i++){
        Q[i][i]   = sg*sg;
        Q[3+i][3+i] = sa*sa;
        Q[6+i][6+i] = 0.0; // not used in F formulation below
        Q[9+i][9+i]  = 2*sbg*sbg/tau_g;
        Q[12+i][12+i]= 2*sba*sba/tau_a;
    }
    return Q;
}

// ── Main ESKF run ─────────────────────────────────────────────────────────────
inline FilterResult run_eskf(const AlignedData& ad){
    FilterResult res;
    res.name="ESKF";
    int N=(int)ad.imu_t.size();
    res.pos.resize(N);
    res.quat.resize(N);
    res.timing_ms.resize(N,0.0);

    ESKFState s=eskf_init(ad);
    res.pos[0]=s.p; res.quat[0]=s.q;

    static const Vec3 g_world={0,0,-9.81};
    double tau_g=100,tau_a=100;
    Mat15 Q=make_Q_eskf();

    double sg=0.0002,sa=0.002,sbg=0.00001,sba=0.0001;

    // Camera index
    int cam_idx=0;
    int N_cam=(int)ad.cam_t.size();

    double sigma_pos_vo=0.15;
    double sigma_ori_vo=0.5*M_PI/180.0;

    RNG rng;

    for(int idx=1;idx<N;idx++){
        auto t0=std::chrono::high_resolution_clock::now();

        double dt=ad.imu_t[idx]-ad.imu_t[idx-1];
        if(dt<=0 || dt>0.05){
            res.pos[idx]=res.pos[idx-1]; res.quat[idx]=res.quat[idx-1];
            continue;
        }

        Vec3 omega=ad.gyro[idx];
        Vec3 acc=ad.accel[idx];
        Vec3 omega_c=omega-s.b_g;
        Vec3 acc_c=acc-s.b_a;

        // --- Propagate orientation ---
        double theta=norm(omega_c)*dt;
        Mat3 dR=(theta<1e-8)? eye3()+skew(omega_c*dt) : expm_so3(omega_c*dt);
        s.R=s.R*dR;
        s.q=rotm2quat(s.R);

        // --- Propagate position/velocity ---
        Vec3 a_world=s.R*acc_c+g_world;
        s.p=s.p+s.v*dt;
        s.v=s.v+a_world*dt;

        // --- Covariance propagation ---
        Mat15 F;  // zero-initialized by default (Mat15 ctor zeros)
        // F(1:3,1:3) = -skew(omega_c)
        Mat3 sk_w=skew(omega_c);
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) F[i][j]=-sk_w[i][j];
        // F(1:3,10:12) = -I   (bias gyro → orientation)
        for(int i=0;i<3;i++) F[i][9+i]=-1;
        // F(4:6,7:9) = I
        for(int i=0;i<3;i++) F[3+i][6+i]=1;
        // F(7:9,1:3) = -R*skew(acc_c)
        Mat3 Rsa=s.R*skew(acc_c);
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) F[6+i][j]=-Rsa[i][j];
        // F(7:9,13:15) = -R
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) F[6+i][12+j]=-s.R[i][j];
        // F(10:12,10:12) = -1/tau_g
        for(int i=0;i<3;i++) F[9+i][9+i]=-1/tau_g;
        // F(13:15,13:15) = -1/tau_a
        for(int i=0;i<3;i++) F[12+i][12+i]=-1/tau_a;

        // Fd = I + F*dt
        Mat15 Fd=eye15();
        for(int i=0;i<15;i++) for(int j=0;j<15;j++) Fd[i][j]+=F[i][j]*dt;

        // ==============================================================
        // FIXED: Correct discrete-time process noise covariance Qd
        // ==============================================================
        Mat15 Qd = {};  // zero-initialized

        double dt2 = dt * dt;
        double dt3 = dt2 * dt;

        // 1. Orientation error: gyro noise
        for(int i = 0; i < 3; i++) Qd[i][i] = sg * sg * dt;

        // 2. Velocity: acceleration noise (single integral)
        for(int i = 0; i < 3; i++) Qd[6+i][6+i] = sa * sa * dt;

        // 3. Position: acceleration noise (double integral) - KEY FIX
        for(int i = 0; i < 3; i++) Qd[3+i][3+i] = sa * sa * dt3 / 3.0;

        // 4. Position-velocity cross-correlation (from same acceleration noise)
        for(int i = 0; i < 3; i++) {
            Qd[3+i][6+i] = sa * sa * dt2 / 2.0;
            Qd[6+i][3+i] = Qd[3+i][6+i];
        }

        // 5. Gyro bias random walk
        for(int i = 0; i < 3; i++) Qd[9+i][9+i] = sbg * sbg * dt;

        // 6. Accel bias random walk
        for(int i = 0; i < 3; i++) Qd[12+i][12+i] = sba * sba * dt;

        // Update covariance with correct Qd (replaces mat15_scale(dt, Q))
        s.P = mat15_symmetrize(mat15_add(mat15_mul(mat15_mul(Fd,s.P),mat15_T(Fd)), Qd));
        // ==============================================================

        // --- ZUPT ---
        apply_zupt(s,omega,acc);
        // --- Gravity alignment ---
        apply_gravity_align(s,acc);

        // --- Camera update ---
        double t_curr=ad.imu_t[idx];
        if(cam_idx<N_cam && t_curr>=ad.cam_t[cam_idx]){
            Vec3 vo_p=ad.cam_pos[cam_idx]+sigma_pos_vo*rng.randn3();
            Mat3 gt_R=quat2rotm(ad.cam_quat[cam_idx]);
            Mat3 vo_R=gt_R*expm_so3(sigma_ori_vo*rng.randn3());

            // Position update
            {
                H3 H{};
                H[0][3]=1; H[1][4]=1; H[2][5]=1;
                double sig2=sigma_pos_vo*sigma_pos_vo;
                Mat3 S=mat_HPH(H,s.P);
                for(int i=0;i<3;i++) S[i][i]+=sig2;
                PH3 PH=mat_PH(s.P,H);
                PH3 K=mat_Kgain(PH,inv3(S));
                Vec3 innov=vo_p-s.p;
                Vec15 dx=mat_Kinnov(K,innov);
                s.p[0]+=dx[3]; s.p[1]+=dx[4]; s.p[2]+=dx[5];
                s.v[0]+=dx[6]; s.v[1]+=dx[7]; s.v[2]+=dx[8];
                s.b_g[0]+=dx[9]; s.b_g[1]+=dx[10]; s.b_g[2]+=dx[11];
                s.b_a[0]+=dx[12]; s.b_a[1]+=dx[13]; s.b_a[2]+=dx[14];
                // clamp biases
                for(int i=0;i<3;i++){
                    s.b_g[i]=std::max(std::min(s.b_g[i],0.01),-0.01);
                    s.b_a[i]=std::max(std::min(s.b_a[i],0.1),-0.1);
                }
                s.P=joseph_update(s.P,K,H,sig2);
            }

            // Orientation update (Lie group)
            {
                Mat3 R_err=transpose(s.R)*vo_R;
                Vec3 ang_err=rotm_to_axis_angle(R_err);
                H3 H{};
                H[0][0]=1; H[1][1]=1; H[2][2]=1;
                double sig2=sigma_ori_vo*sigma_ori_vo;
                Mat3 S=mat_HPH(H,s.P);
                for(int i=0;i<3;i++) S[i][i]+=sig2;
                PH3 PH=mat_PH(s.P,H);
                PH3 K=mat_Kgain(PH,inv3(S));
                Vec15 dx=mat_Kinnov(K,ang_err);
                Vec3 dphi={dx[0],dx[1],dx[2]};
                s.R=s.R*expm_so3(dphi);
                s.q=rotm2quat(s.R);
                s.P=joseph_update(s.P,K,H,sig2);
            }

            cam_idx++;
        }

        res.pos[idx]=s.p;
        res.quat[idx]=s.q;

        auto t1=std::chrono::high_resolution_clock::now();
        res.timing_ms[idx]=std::chrono::duration<double,std::milli>(t1-t0).count();

        if(idx%5000==0)
            std::printf("  ESKF %d/%d  %.3f ms\n",idx,N,res.timing_ms[idx]);
    }

    compute_rmse(res,ad);
    std::printf("\n=== ESKF RESULTS ===\n");
    std::printf("Position RMSE : %.3f m\n",res.pos_rmse);
    std::printf("Orientation RMSE: %.2f deg\n",res.ori_rmse_deg);
    std::printf("Avg time      : %.4f ms\n",res.avg_time_ms);
    return res;
}