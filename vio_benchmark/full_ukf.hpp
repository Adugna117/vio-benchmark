#pragma once
// =============================================================================
// full_ukf.hpp  –  Full 15-state Unscented Kalman Filter
//                  Port of run_full_ukf.m
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
// UKF sigma-point / weight parameters  (standard Merwe α=0.1, β=2, κ=0)
// n=15  →  2n+1=31 sigma points
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int UKF_N = 15;
static constexpr int UKF_L = 2*UKF_N+1;  // 31

struct UKFWeights {
    double Wm[UKF_L]{};
    double Wc[UKF_L]{};
};

inline UKFWeights make_weights(double alpha=0.1,double beta=2.0,double kappa=0.0){
    double n=UKF_N;
    double lam=alpha*alpha*(n+kappa)-n;
    UKFWeights w;
    w.Wm[0]=lam/(n+lam);
    w.Wc[0]=lam/(n+lam)+(1-alpha*alpha+beta);
    for(int i=1;i<UKF_L;i++){
        w.Wm[i]=1.0/(2*(n+lam));
        w.Wc[i]=1.0/(2*(n+lam));
    }
    return w;
}

// ── Compact 15-dim state vector ───────────────────────────────────────────────
// Layout: [phi(0:2), p(3:5), v(6:8), bg(9:11), ba(12:14)]
// NOTE: orientation stored as error-state phi; nominal quaternion tracked
//       separately and applied after.

struct UKFFullState {
    Vec3  p{},v{},b_g{},b_a{};
    Vec4  q{0,0,0,1};
    Mat3  R=eye3();
    Mat15 P;   // 15×15 covariance
};

inline UKFFullState ukf_full_init(const AlignedData& ad){
    UKFFullState s;
    s.p=ad.gt_pos[0];
    s.q=qnorm(ad.gt_quat[0]);
    s.R=quat2rotm(s.q);
    // covariance
    for(int i=0;i<3;i++){
        s.P[i][i]    = (0.5*M_PI/180)*(0.5*M_PI/180);
        s.P[3+i][3+i]= 0.01*0.01;
        s.P[6+i][6+i]= 0.05*0.05;
        s.P[9+i][9+i] = 0.001*0.001;
        s.P[12+i][12+i]=0.01*0.01;
    }
    return s;
}

// ── Cholesky decomposition of 15×15 SPD matrix → lower triangular ─────────────
inline Mat15 chol15(const Mat15& A){
    Mat15 L;
    for(int i=0;i<15;i++){
        for(int j=0;j<=i;j++){
            double s=A[i][j];
            for(int k=0;k<j;k++) s-=L[i][k]*L[j][k];
            if(i==j){
                L[i][j]=(s>0)?std::sqrt(s):1e-9;
            } else {
                L[i][j]=(std::abs(L[j][j])>1e-14)?s/L[j][j]:0.0;
            }
        }
    }
    return L;
}

// ── Generate 31 sigma points from mean (zero) + P ────────────────────────────
// Returns columns stored as rows: sigma[k][i] = k-th sigma point, i-th dim
using SigmaMat = std::array<std::array<double,UKF_N>,UKF_L>;

inline SigmaMat sigma_points(const Mat15& P, double alpha=0.1, double kappa=0.0){
    double n=UKF_N;
    double lam=alpha*alpha*(n+kappa)-n;
    double scale=std::sqrt(n+lam);

    // Scale each col of cholesky by scale
    Mat15 L=chol15(P);
    SigmaMat S{};
    // S[0] = mean = 0
    for(int k=0;k<UKF_N;k++){
        // S[k+1]  = +scale * col_k(L)
        // S[k+1+N]= -scale * col_k(L)
        for(int i=0;i<UKF_N;i++){
            S[k+1][i]       = +scale*L[i][k];
            S[k+1+UKF_N][i] = -scale*L[i][k];
        }
    }
    return S;
}

// ── Process model:  propagate one sigma point (15-dim) ────────────────────────
// xi = error-state sigma point  (phi,p,v,bg,ba)
// nominal state (R,p,v,bg,ba) passed via references
// Returns propagated sigma point
inline std::array<double,UKF_N> ukf_process(
    const std::array<double,UKF_N>& xi,
    const Vec3& p_nom, const Vec3& v_nom,
    const Vec4& q_nom, const Mat3& R_nom,
    const Vec3& bg_nom, const Vec3& ba_nom,
    Vec3 omega_raw, Vec3 acc_raw, double dt)
{
    static const Vec3 gw={0,0,-9.81};

    // Recover perturbed state
    Vec3 phi={xi[0],xi[1],xi[2]};
    Vec3 p={p_nom[0]+xi[3], p_nom[1]+xi[4], p_nom[2]+xi[5]};
    Vec3 v={v_nom[0]+xi[6], v_nom[1]+xi[7], v_nom[2]+xi[8]};
    Vec3 bg={bg_nom[0]+xi[9],  bg_nom[1]+xi[10], bg_nom[2]+xi[11]};
    Vec3 ba={ba_nom[0]+xi[12], ba_nom[1]+xi[13], ba_nom[2]+xi[14]};

    // Perturb rotation
    Mat3 R=R_nom*expm_so3(phi);

    Vec3 omega_c=omega_raw-bg;
    Vec3 acc_c=acc_raw-ba;

    // Propagate
    Mat3 dR=expm_so3(omega_c*dt);
    Mat3 R_new=R*dR;
    Vec3 a_world=R*acc_c+gw;
    Vec3 v_new=v+a_world*dt;
    Vec3 p_new=p+v*dt+0.5*a_world*dt*dt;
    Vec3 bg_new=bg;
    Vec3 ba_new=ba;

    // New phi relative to propagated nominal
    // phi_new ≈ log(R_nom_new' * R_new)
    // R_nom_new = R_nom * expm(omega_c_nom * dt)  but we use R_new itself
    // → just return residual = 0 in orientation (nominal tracks mean)
    std::array<double,UKF_N> xo{};
    // phi_out = log( R_new * R_nom_new^{-1} ) but since we track around nominal
    // just return the deltas
    xo[0]=0; xo[1]=0; xo[2]=0; // orientation residual (absorbed into nominal)
    xo[3]=p_new[0]; xo[4]=p_new[1]; xo[5]=p_new[2];
    xo[6]=v_new[0]; xo[7]=v_new[1]; xo[8]=v_new[2];
    xo[9]=bg_new[0]; xo[10]=bg_new[1]; xo[11]=bg_new[2];
    xo[12]=ba_new[0]; xo[13]=ba_new[1]; xo[14]=ba_new[2];
    return xo;
}

// ── ZUPT for UKF (reuse ESKF linear approach) ────────────────────────────────
inline void ukf_zupt(UKFFullState& s, Vec3 omega_raw, Vec3 acc_raw){
    bool stat=(norm(omega_raw)<0.01)&&(std::abs(norm(acc_raw)-9.81)<0.05);
    if(!stat) return;
    H3 H{};
    H[0][6]=1; H[1][7]=1; H[2][8]=1;
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

inline void ukf_gravity(UKFFullState& s, Vec3 acc_raw){
    if(std::abs(norm(acc_raw)-9.81)>=0.1) return;
    Vec3 g_meas=normalized(acc_raw);
    Vec3 g_exp={-s.R[0][2],-s.R[1][2],-s.R[2][2]};
    Vec3 innov=g_meas-g_exp;
    H3 H{};
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

// ── Process noise for UKF ────────────────────────────────────────────────────
inline Mat15 make_Q_ukf(){
    Mat15 Q;
    for(int i=0;i<3;i++){
        Q[i][i]     = 0.01*0.01;
        Q[3+i][3+i] = 0.01*0.01;
        Q[6+i][6+i] = 0.001*0.001;
        Q[9+i][9+i] = 0.0001*0.0001;
        Q[12+i][12+i]= 0.001*0.001;
    }
    return Q;
}

// ── Main Full UKF run ─────────────────────────────────────────────────────────
inline FilterResult run_full_ukf(const AlignedData& ad){
    FilterResult res;
    res.name="Full_UKF";
    int N=(int)ad.imu_t.size();
    res.pos.resize(N);
    res.quat.resize(N);
    res.timing_ms.resize(N,0.0);

    UKFFullState s=ukf_full_init(ad);
    res.pos[0]=s.p; res.quat[0]=s.q;

    Mat15 Q=make_Q_ukf();
    UKFWeights W=make_weights();

    int cam_idx=0;
    int N_cam=(int)ad.cam_t.size();
    double sigma_pos_vo=0.15;
    double sigma_ori_vo=0.5*M_PI/180.0;

    RNG rng;

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

        // ─── UKF Predict ─────────────────────────────────────────────────────
        // 1. Generate sigma points around ZERO (error state)
        SigmaMat SP=sigma_points(s.P);

        // 2. Propagate each sigma point through nonlinear model
        SigmaMat SP_pred{};
        static const Vec3 gw={0,0,-9.81};

        // Nominal propagation (sigma[0])
        Mat3 dR_nom=expm_so3(omega_c*dt);
        Mat3 R_pred_nom=s.R*dR_nom;
        Vec3 a_w=s.R*acc_c+gw;
        Vec3 v_pred_nom=s.v+a_w*dt;
        Vec3 p_pred_nom=s.p+s.v*dt+0.5*a_w*dt*dt;
        Vec3 bg_pred=s.b_g, ba_pred=s.b_a;
        Vec4 q_pred_nom=rotm2quat(R_pred_nom);

        // SP[0] stays zero (mean)
        // For SP[1..30]: propagate perturbation
        for(int k=1;k<UKF_L;k++){
            // Perturb: extract phi, then compute perturbed rotation
            Vec3 phi_k={SP[k][0],SP[k][1],SP[k][2]};
            Vec3 dp_k={SP[k][3],SP[k][4],SP[k][5]};
            Vec3 dv_k={SP[k][6],SP[k][7],SP[k][8]};
            Vec3 dbg_k={SP[k][9],SP[k][10],SP[k][11]};
            Vec3 dba_k={SP[k][12],SP[k][13],SP[k][14]};

            Vec3 bg_k=s.b_g+dbg_k;
            Vec3 ba_k=s.b_a+dba_k;
            Vec3 om_k=omega-bg_k;
            Vec3 ac_k=acc-ba_k;

            Mat3 R_k=s.R*expm_so3(phi_k);
            Mat3 dR_k=expm_so3(om_k*dt);
            Mat3 R_new_k=R_k*dR_k;

            Vec3 a_k=R_k*ac_k+gw;
            Vec3 v_new_k={s.v[0]+dv_k[0]+a_k[0]*dt,
                          s.v[1]+dv_k[1]+a_k[1]*dt,
                          s.v[2]+dv_k[2]+a_k[2]*dt};
            Vec3 p_new_k={s.p[0]+dp_k[0]+(s.v[0]+dv_k[0])*dt+0.5*a_k[0]*dt*dt,
                          s.p[1]+dp_k[1]+(s.v[1]+dv_k[1])*dt+0.5*a_k[1]*dt*dt,
                          s.p[2]+dp_k[2]+(s.v[2]+dv_k[2])*dt+0.5*a_k[2]*dt*dt};

            // phi residual relative to nominal propagated
            Vec3 phi_res=rotm_to_axis_angle(transpose(R_pred_nom)*R_new_k);

            SP_pred[k][0]=phi_res[0]; SP_pred[k][1]=phi_res[1]; SP_pred[k][2]=phi_res[2];
            SP_pred[k][3]=p_new_k[0]-p_pred_nom[0];
            SP_pred[k][4]=p_new_k[1]-p_pred_nom[1];
            SP_pred[k][5]=p_new_k[2]-p_pred_nom[2];
            SP_pred[k][6]=v_new_k[0]-v_pred_nom[0];
            SP_pred[k][7]=v_new_k[1]-v_pred_nom[1];
            SP_pred[k][8]=v_new_k[2]-v_pred_nom[2];
            for(int d=0;d<3;d++){
                SP_pred[k][9+d] =dbg_k[d];
                SP_pred[k][12+d]=dba_k[d];
            }
        }

        // 3. Predicted mean (should be ~zero for error state)
        std::array<double,UKF_N> xm{};
        for(int k=0;k<UKF_L;k++)
            for(int d=0;d<UKF_N;d++) xm[d]+=W.Wm[k]*SP_pred[k][d];

        // 4. Predicted covariance + Q
        Mat15 P_pred{};
        for(int k=0;k<UKF_L;k++){
            std::array<double,UKF_N> dx;
            for(int d=0;d<UKF_N;d++) dx[d]=SP_pred[k][d]-xm[d];
            for(int i=0;i<UKF_N;i++) for(int j=0;j<UKF_N;j++)
                P_pred[i][j]+=W.Wc[k]*dx[i]*dx[j];
        }
        P_pred=mat15_add(P_pred,mat15_scale(dt,Q));
        P_pred=mat15_symmetrize(P_pred);

        // Update nominal state
        s.R=R_pred_nom; s.q=q_pred_nom;
        s.p=p_pred_nom; s.v=v_pred_nom;
        s.P=P_pred;

        // ─── ZUPT + Gravity ────────────────────────────────────────────────
        ukf_zupt(s,omega,acc);
        ukf_gravity(s,acc);

        // ─── Camera update ─────────────────────────────────────────────────
        double t_curr=ad.imu_t[idx];
        if(cam_idx<N_cam && t_curr>=ad.cam_t[cam_idx]){
            Vec3 vo_p=ad.cam_pos[cam_idx]+sigma_pos_vo*rng.randn3();
            Mat3 gt_R=quat2rotm(ad.cam_quat[cam_idx]);
            Mat3 vo_R=gt_R*expm_so3(sigma_ori_vo*rng.randn3());

            // Combined 6-DOF update (orientation + position)
            // z = [angle_error(3); pos_innov(3)]
            Mat3 R_err=transpose(s.R)*vo_R;
            Vec3 ang_err=rotm_to_axis_angle(R_err);
            Vec3 pos_inn=vo_p-s.p;

            H6 H6m{};
            // H_ori: rows 0-2, cols 0-2 = I
            H6m[0][0]=1; H6m[1][1]=1; H6m[2][2]=1;
            // H_pos: rows 3-5, cols 3-5 = I
            H6m[3][3]=1; H6m[4][4]=1; H6m[5][5]=1;

            double so2=sigma_ori_vo*sigma_ori_vo;
            double sp2=sigma_pos_vo*sigma_pos_vo;
            Mat6 S6=mat_HPH6(H6m,s.P);
            for(int i=0;i<3;i++) S6.m[i][i]+=so2;
            for(int i=3;i<6;i++) S6.m[i][i]+=sp2;
            // symmetrize
            for(int i=0;i<6;i++) for(int j=0;j<6;j++)
                S6.m[i][j]=(S6.m[i][j]+S6.m[j][i])/2;
            for(int i=0;i<6;i++) S6.m[i][i]+=1e-6;

            PH6 PH6v=mat_PH6(s.P,H6m);
            Mat6 S6inv=inv6_blockdiag(S6);
            PH6 K6=mat_Kgain6(PH6v,S6inv);

            std::array<double,6> innov6{ang_err[0],ang_err[1],ang_err[2],
                                         pos_inn[0],pos_inn[1],pos_inn[2]};
            Vec15 dx=mat_Kinnov6(K6,innov6);

            // Apply
            Vec3 dphi={dx[0],dx[1],dx[2]};
            s.R=s.R*expm_so3(dphi);
            s.q=rotm2quat(s.R);
            s.p[0]+=dx[3]; s.p[1]+=dx[4]; s.p[2]+=dx[5];
            s.v[0]+=dx[6]; s.v[1]+=dx[7]; s.v[2]+=dx[8];
            s.b_g[0]+=dx[9]; s.b_g[1]+=dx[10]; s.b_g[2]+=dx[11];
            s.b_a[0]+=dx[12]; s.b_a[1]+=dx[13]; s.b_a[2]+=dx[14];
            for(int i=0;i<3;i++){
                s.b_g[i]=std::max(std::min(s.b_g[i],0.01),-0.01);
                s.b_a[i]=std::max(std::min(s.b_a[i],0.1),-0.1);
            }
            s.P=joseph_update6(s.P,K6,H6m,so2,sp2);
            cam_idx++;
        }

        res.pos[idx]=s.p;
        res.quat[idx]=s.q;
        auto t1c=std::chrono::high_resolution_clock::now();
        res.timing_ms[idx]=std::chrono::duration<double,std::milli>(t1c-t0c).count();

        if(idx%5000==0)
            std::printf("  UKF %d/%d  %.3f ms\n",idx,N,res.timing_ms[idx]);
    }

    compute_rmse(res,ad);
    std::printf("\n=== FULL UKF RESULTS ===\n");
    std::printf("Position RMSE : %.3f m\n",res.pos_rmse);
    std::printf("Orientation RMSE: %.2f deg\n",res.ori_rmse_deg);
    std::printf("Avg time      : %.4f ms\n",res.avg_time_ms);
    return res;
}
