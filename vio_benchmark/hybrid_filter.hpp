#pragma once
// =============================================================================
// hybrid_filter.hpp  –  Hybrid ESKF (position/velocity/bias) +
//                       UKF orientation refinement
//                       Port of run_hybrid_filter.m
// =============================================================================
#include <cstdio>
#include <cmath>
#include <array>
#include <vector>
#include <chrono>
#include <random>
#include "math_utils.hpp"
#include "data_io.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Lightweight UKF for quaternion (4-state)
// We track q ∈ S^3,  propagate via IMU omega,  correct with VO quaternion
// ─────────────────────────────────────────────────────────────────────────────
// Add RNG struct if not already present (same as in eskf.hpp)

static constexpr int QUKF_N = 4;
static constexpr int QUKF_L = 2*QUKF_N+1;   // 9 sigma points

struct QUKFWeights {
    double Wm[QUKF_L]{};
    double Wc[QUKF_L]{};
};
inline QUKFWeights make_q_weights(double alpha=0.1,double beta=2.0,double kappa=0.0){
    double n=QUKF_N;
    double lam=alpha*alpha*(n+kappa)-n;
    QUKFWeights w;
    w.Wm[0]=lam/(n+lam);
    w.Wc[0]=lam/(n+lam)+(1-alpha*alpha+beta);
    for(int i=1;i<QUKF_L;i++){
        w.Wm[i]=1.0/(2*(n+lam));
        w.Wc[i]=1.0/(2*(n+lam));
    }
    return w;
}

// 4×4 covariance + 4-vector state
struct QUKF {
    Vec4 q{0,0,0,1};
    double P[4][4]{};
    double Q_proc[4][4]{};
    double R_meas[4][4]{};
    QUKFWeights W;

    void init(Vec4 q0, double proc_noise=0.001, double meas_noise=0.01){
        q=qnorm(q0);
        for(int i=0;i<4;i++){
            P[i][i]=0.01;
            Q_proc[i][i]=proc_noise;
            R_meas[i][i]=meas_noise;
        }
        W=make_q_weights();
    }

    // Cholesky of 4×4
    static void chol4(const double A[4][4], double L[4][4]){
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) L[i][j]=0;
        for(int i=0;i<4;i++){
            for(int j=0;j<=i;j++){
                double s=A[i][j];
                for(int k=0;k<j;k++) s-=L[i][k]*L[j][k];
                L[i][j]=(i==j)?((s>0)?std::sqrt(s):1e-9):(std::abs(L[j][j])>1e-14?s/L[j][j]:0.0);
            }
        }
    }

    // Generate 9 sigma points
    void sigma_pts(double sp[QUKF_L][QUKF_N]){
        double n=QUKF_N;
        double alpha=0.1, kappa=0.0;
        double lam=alpha*alpha*(n+kappa)-n;
        double scale=std::sqrt(n+lam);
        double L[4][4]{};
        chol4(P,L);
        for(int d=0;d<4;d++) sp[0][d]=q[d];
        for(int k=0;k<4;k++){
            for(int d=0;d<4;d++){
                sp[k+1][d]    =q[d]+scale*L[d][k];
                sp[k+1+4][d]  =q[d]-scale*L[d][k];
            }
        }
    }

    // Predict: propagate quaternion through constant model (identity)
    void predict(){
        // Process: q_pred = q  (IMU integration done externally in ESKF)
        // Just inflate covariance
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) P[i][j]+=Q_proc[i][j];
    }

    // Correct with measurement z_q (quaternion from VO)
    void correct(Vec4 z_q){
        // Generate sigma points
        double sp[QUKF_L][QUKF_N]{};
        sigma_pts(sp);

        // Propagate (identity model)
        // Predicted mean
        double xm[4]={};
        for(int k=0;k<QUKF_L;k++)
            for(int d=0;d<4;d++) xm[d]+=W.Wm[k]*sp[k][d];
        // Normalise mean
        double nm=0; for(int d=0;d<4;d++) nm+=xm[d]*xm[d]; nm=std::sqrt(nm);
        for(int d=0;d<4;d++) xm[d]/=nm;

        // Predicted covariance P_xx
        double Pxx[4][4]{};
        for(int k=0;k<QUKF_L;k++){
            double dx[4]; for(int d=0;d<4;d++) dx[d]=sp[k][d]-xm[d];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++) Pxx[i][j]+=W.Wc[k]*dx[i]*dx[j];
        }
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) Pxx[i][j]+=Q_proc[i][j];

        // Measurement prediction = sigma point (identity h(x)=x)
        // Innovation covariance S = Pxx + R
        double S[4][4]{};
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) S[i][j]=Pxx[i][j]+R_meas[i][j];
        // Symmetrize + regularise
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) S[i][j]=(S[i][j]+S[j][i])/2;
        for(int i=0;i<4;i++) S[i][i]+=1e-8;

        // Cross covariance P_xz = Pxx (since h=I)
        // K = Pxx * S^{-1}
        // Gauss-Jordan invert S into Sinv
        double a[4][8]{};
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++) a[i][j]=S[i][j];
            a[i][4+i]=1.0;
        }
        for(int col=0;col<4;col++){
            int piv=col;
            for(int row=col+1;row<4;row++) if(std::abs(a[row][col])>std::abs(a[piv][col])) piv=row;
            for(int c=0;c<8;c++) std::swap(a[col][c],a[piv][c]);
            double sc=a[col][col]; if(std::abs(sc)<1e-14) sc=1e-14;
            for(int c=0;c<8;c++) a[col][c]/=sc;
            for(int row=0;row<4;row++) if(row!=col){
                double f=a[row][col];
                for(int c=0;c<8;c++) a[row][c]-=f*a[col][c];
            }
        }
        double Sinv[4][4]{};
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) Sinv[i][j]=a[i][4+j];

        // K = Pxx * Sinv
        double K[4][4]{};
        for(int i=0;i<4;i++) for(int j=0;j<4;j++)
            for(int k2=0;k2<4;k2++) K[i][j]+=Pxx[i][k2]*Sinv[k2][j];

        // Innovation
        double z[4]={z_q[0],z_q[1],z_q[2],z_q[3]};
        double innov[4]; for(int d=0;d<4;d++) innov[d]=z[d]-xm[d];

        // Update q
        double qn[4]; for(int d=0;d<4;d++) qn[d]=xm[d];
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) qn[i]+=K[i][j]*innov[j];
        nm=0; for(int d=0;d<4;d++) nm+=qn[d]*qn[d]; nm=std::sqrt(nm);
        for(int d=0;d<4;d++) q[d]=qn[d]/nm;

        // Update P = (I-K)*Pxx*(I-K)' + K*R*K'  (Joseph)
        double IK[4][4]{};
        for(int i=0;i<4;i++) IK[i][i]=1;
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) IK[i][j]-=K[i][j];

        double tmp[4][4]{};
        for(int i=0;i<4;i++) for(int k2=0;k2<4;k2++) for(int j=0;j<4;j++)
            tmp[i][j]+=IK[i][k2]*Pxx[k2][j];
        double Pn[4][4]{};
        for(int i=0;i<4;i++) for(int k2=0;k2<4;k2++) for(int j=0;j<4;j++)
            Pn[i][j]+=tmp[i][k2]*IK[j][k2]; // *(IK^T)
        // + K*R*K'
        for(int i=0;i<4;i++) for(int k2=0;k2<4;k2++) for(int j=0;j<4;j++)
            Pn[i][j]+=K[i][k2]*R_meas[k2][k2]*K[j][k2];
        // symmetrize + copy
        for(int i=0;i<4;i++) for(int j=0;j<4;j++)
            P[i][j]=(Pn[i][j]+Pn[j][i])/2;
        for(int i=0;i<4;i++) P[i][i]+=1e-8;
    }
};

// ── Hybrid filter state ────────────────────────────────────────────────────────
struct HybridState {
    Vec3  p{},v{},b_g{},b_a{};
    Vec4  q{0,0,0,1};
    Mat3  R=eye3();
    Mat15 P;
    QUKF  qukf;
};

inline HybridState hybrid_init(const AlignedData& ad){
    HybridState s;
    s.p=ad.gt_pos[0];
    s.q=qnorm(ad.gt_quat[0]);
    s.R=quat2rotm(s.q);
    for(int i=0;i<3;i++){
        s.P[i][i]    =(0.5*M_PI/180)*(0.5*M_PI/180);
        s.P[3+i][3+i]=0.01*0.01;
        s.P[6+i][6+i]=0.05*0.05;
        s.P[9+i][9+i] =0.001*0.001;
        s.P[12+i][12+i]=0.01*0.01;
    }
    s.qukf.init(s.q,0.001,0.01);
    return s;
}

inline Mat15 make_Q_hybrid(){
    double sg=0.0002,sa=0.002,sbg=0.00001,sba=0.0001,tau_g=100,tau_a=100;
    Mat15 Q;
    for(int i=0;i<3;i++){
        Q[i][i]=sg*sg; Q[3+i][3+i]=sa*sa;
        Q[9+i][9+i]=2*sbg*sbg/tau_g; Q[12+i][12+i]=2*sba*sba/tau_a;
    }
    return Q;
}

inline FilterResult run_hybrid_filter(const AlignedData& ad, uint64_t seed = 42){
    FilterResult res;
    res.name="Hybrid";
    int N=(int)ad.imu_t.size();
    res.pos.resize(N); res.quat.resize(N); res.timing_ms.resize(N,0.0);

    HybridState s=hybrid_init(ad);
    res.pos[0]=s.p; res.quat[0]=s.q;

    double tau_g=100, tau_a=100;
    Mat15 Q=make_Q_hybrid();

    int cam_idx=0, N_cam=(int)ad.cam_t.size();
    double sigma_pos_vo=0.15;
    double sigma_ori_vo=0.5*M_PI/180.0;

RNG rng;  // seed is fixed at 42 inside the struct

    for(int idx=1;idx<N;idx++){
        auto t0c=std::chrono::high_resolution_clock::now();

        double dt=ad.imu_t[idx]-ad.imu_t[idx-1];
        if(dt<=0||dt>0.05){
            res.pos[idx]=res.pos[idx-1]; res.quat[idx]=res.quat[idx-1];
            continue;
        }

        Vec3 omega=ad.gyro[idx];
        Vec3 acc=ad.accel[idx];
        Vec3 omega_c=omega-s.b_g;
        Vec3 acc_c=acc-s.b_a;

        // ─── ESKF propagation ────────────────────────────────────────────────
        Mat3 dR=expm_so3(omega_c*dt);
        s.R=s.R*dR;
        s.q=rotm2quat(s.R);

        static const Vec3 gw={0,0,-9.81};
        Vec3 a_world=s.R*acc_c+gw;
        s.v=s.v+a_world*dt;
        s.p=s.p+s.v*dt;

        // Covariance propagation
        Mat15 F;
        Mat3 sk_w=skew(omega_c);
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) F[i][j]=-sk_w[i][j];
        for(int i=0;i<3;i++) F[i][9+i]=-1;
        for(int i=0;i<3;i++) F[3+i][6+i]=1;
        Mat3 Rsa=s.R*skew(acc_c);
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) F[6+i][j]=-Rsa[i][j];
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) F[6+i][12+j]=-s.R[i][j];
        for(int i=0;i<3;i++) F[9+i][9+i]=-1/tau_g;
        for(int i=0;i<3;i++) F[12+i][12+i]=-1/tau_a;

        Mat15 Fd=eye15();
        for(int i=0;i<15;i++) for(int j=0;j<15;j++) Fd[i][j]+=F[i][j]*dt;
        // ==============================================================
// Correct discrete-time process noise covariance Qd
// ==============================================================
Mat15 Qd = {};

double dt2 = dt * dt;
double dt3 = dt2 * dt;

// Noise parameters (same as ESKF)
double sg = 0.0002;
double sa = 0.002;
double sbg = 0.00001;
double sba = 0.0001;

// 1. Orientation error: gyro noise
for(int i = 0; i < 3; i++) Qd[i][i] = sg * sg * dt;

// 2. Velocity: acceleration noise (single integral)
for(int i = 0; i < 3; i++) Qd[6+i][6+i] = sa * sa * dt;

// 3. Position: acceleration noise (double integral)
for(int i = 0; i < 3; i++) Qd[3+i][3+i] = sa * sa * dt3 / 3.0;

// 4. Position-velocity cross-correlation
for(int i = 0; i < 3; i++) {
    Qd[3+i][6+i] = sa * sa * dt2 / 2.0;
    Qd[6+i][3+i] = Qd[3+i][6+i];
}

// 5. Gyro bias random walk
for(int i = 0; i < 3; i++) Qd[9+i][9+i] = sbg * sbg * dt;

// 6. Accel bias random walk
for(int i = 0; i < 3; i++) Qd[12+i][12+i] = sba * sba * dt;

// Update covariance with correct Qd
s.P = mat15_symmetrize(mat15_add(mat15_mul(mat15_mul(Fd,s.P),mat15_T(Fd)), Qd));

        // ─── ZUPT ────────────────────────────────────────────────────────────
        {
            bool stat=(norm(omega)<0.01)&&(std::abs(norm(acc)-9.81)<0.05);
            if(stat){
                H3 H{}; H[0][6]=1; H[1][7]=1; H[2][8]=1;
                double sig2=0.01*0.01;
                Mat3 S=mat_HPH(H,s.P);
                for(int i=0;i<3;i++) S[i][i]+=sig2+1e-6;
                PH3 PH=mat_PH(s.P,H);
                PH3 K=mat_Kgain(PH,inv3(S));
                for(int i=0;i<15;i++) for(int j=0;j<3;j++) K[i][j]*=0.3;
                Vec3 innov={-s.v[0],-s.v[1],-s.v[2]};
                Vec15 dx=mat_Kinnov(K,innov);
                s.v[0]+=dx[6]; s.v[1]+=dx[7]; s.v[2]+=dx[8];
                s.P=joseph_update(s.P,K,H,sig2);
            }
        }
        // ─── Gravity alignment ───────────────────────────────────────────────
        {
            if(std::abs(norm(acc)-9.81)<0.1){
                Vec3 g_meas=normalized(acc);
                Vec3 g_exp={-s.R[0][2],-s.R[1][2],-s.R[2][2]};
                Vec3 innov=g_meas-g_exp;
                H3 H{}; Mat3 sk=skew(g_exp);
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
        }

        // ─── Sanity check quaternion ─────────────────────────────────────────
        double qn=norm4(s.q);
        if(std::isnan(qn)||std::isinf(qn)||std::abs(qn-1)>0.1){
            s.q=qnorm(ad.gt_quat[idx]);
            s.R=quat2rotm(s.q);
            s.p=ad.gt_pos[idx]; s.v={0,0,0};
            for(int i=0;i<15;i++) for(int j=0;j<15;j++) s.P[i][j]=(i==j?1e-4:0);
        } else {
            s.q=qnorm(s.q);
        }

        // ─── UKF predict (orientation only) ─────────────────────────────────
        s.qukf.q=s.q;
        s.qukf.predict();

        // ─── Camera update ───────────────────────────────────────────────────
        double t_curr=ad.imu_t[idx];
        if(cam_idx<N_cam && t_curr>=ad.cam_t[cam_idx]){
            Vec3 vo_p=ad.cam_pos[cam_idx]+sigma_pos_vo*rng.randn3();
            Mat3 gt_R=quat2rotm(ad.cam_quat[cam_idx]);
            Mat3 vo_R=gt_R*expm_so3(sigma_ori_vo*rng.randn3());
            Vec4 vo_q=rotm2quat(vo_R);

            // UKF correct orientation
            s.qukf.correct(vo_q);
            s.q=qnorm(s.qukf.q);
            s.R=quat2rotm(s.q);

            // ESKF correct position
            {
                H3 H{}; H[0][3]=1; H[1][4]=1; H[2][5]=1;
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
                for(int i=0;i<3;i++){
                    s.b_g[i]=std::max(std::min(s.b_g[i],0.01),-0.01);
                    s.b_a[i]=std::max(std::min(s.b_a[i],0.1),-0.1);
                }
                s.P=joseph_update(s.P,K,H,sig2);
            }
            cam_idx++;
        }

        res.pos[idx]=s.p;
        res.quat[idx]=s.q;
        auto t1c=std::chrono::high_resolution_clock::now();
        res.timing_ms[idx]=std::chrono::duration<double,std::milli>(t1c-t0c).count();

        if(idx%5000==0)
            std::printf("  Hybrid %d/%d  %.3f ms\n",idx,N,res.timing_ms[idx]);
    }

    compute_rmse(res,ad);
    std::printf("\n=== HYBRID FILTER RESULTS ===\n");
    std::printf("Position RMSE : %.3f m\n",res.pos_rmse);
    std::printf("Orientation RMSE: %.2f deg\n",res.ori_rmse_deg);
    std::printf("Avg time      : %.4f ms\n",res.avg_time_ms);
    return res;
}
