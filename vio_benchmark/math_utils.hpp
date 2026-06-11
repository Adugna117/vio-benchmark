#pragma once
// =============================================================================
// math_utils.hpp  –  Eigen-free fixed-size matrix/quaternion helpers
//                    targeting C++17 / MSYS2 UCRT64 / GCC
// =============================================================================
//#define _USE_MATH_DEFINES
#include <cmath>
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

// ── Typedefs ─────────────────────────────────────────────────────────────────
using Vec3  = std::array<double,3>;
using Vec4  = std::array<double,4>;
using Mat3  = std::array<std::array<double,3>,3>;
using Mat4  = std::array<std::array<double,4>,4>;

// For 15-state filter we use a plain heap array wrapper
struct Vec15 { double v[15]{}; double& operator[](int i){return v[i];} double operator[](int i)const{return v[i];} };
struct Mat15 {
    double m[15][15]{};
    double* operator[](int i){return m[i];}
    const double* operator[](int i)const{return m[i];}
};

// ── Vec3 ops ─────────────────────────────────────────────────────────────────
inline Vec3 operator+(Vec3 a, Vec3 b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline Vec3 operator-(Vec3 a, Vec3 b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline Vec3 operator*(double s, Vec3 a){return {s*a[0],s*a[1],s*a[2]};}
inline Vec3 operator*(Vec3 a, double s){return s*a;}
inline Vec3& operator+=(Vec3& a, Vec3 b){a[0]+=b[0];a[1]+=b[1];a[2]+=b[2];return a;}
inline Vec3& operator-=(Vec3& a, Vec3 b){a[0]-=b[0];a[1]-=b[1];a[2]-=b[2];return a;}
inline double dot(Vec3 a, Vec3 b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline double norm(Vec3 a){return std::sqrt(dot(a,a));}
inline Vec3 normalized(Vec3 a){double n=norm(a); return (n<1e-12)?Vec3{0,0,0}:(1.0/n)*a;}
inline Vec3 cross(Vec3 a, Vec3 b){return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};}

// ── Mat3 ops ─────────────────────────────────────────────────────────────────
inline Mat3 eye3(){return {{{1,0,0},{0,1,0},{0,0,1}}};}

inline Mat3 operator+(Mat3 A, Mat3 B){
    Mat3 C;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) C[i][j]=A[i][j]+B[i][j];
    return C;
}
inline Mat3 operator-(Mat3 A, Mat3 B){
    Mat3 C;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) C[i][j]=A[i][j]-B[i][j];
    return C;
}
inline Mat3 operator*(double s, Mat3 A){
    Mat3 C;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) C[i][j]=s*A[i][j];
    return C;
}
inline Mat3 operator*(Mat3 A, Mat3 B){
    Mat3 C={};
    for(int i=0;i<3;i++) for(int k=0;k<3;k++) for(int j=0;j<3;j++)
        C[i][j]+=A[i][k]*B[k][j];
    return C;
}
inline Vec3 operator*(Mat3 A, Vec3 v){
    return {A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2],
            A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2],
            A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2]};
}
inline Mat3 transpose(Mat3 A){
    Mat3 T;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) T[i][j]=A[j][i];
    return T;
}
inline double trace(Mat3 A){return A[0][0]+A[1][1]+A[2][2];}

// ── SO3 / Quaternion ──────────────────────────────────────────────────────────
// Convention: q = [qx, qy, qz, qw]  (matches MATLAB code)

inline double norm4(Vec4 q){return std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);}
inline Vec4 qnorm(Vec4 q){double n=norm4(q); return {q[0]/n,q[1]/n,q[2]/n,q[3]/n};}

// skew-symmetric matrix
inline Mat3 skew(Vec3 v){
    return {{{0,-v[2],v[1]},{v[2],0,-v[0]},{-v[1],v[0],0}}};
}

// quaternion → rotation matrix  [qx qy qz qw]
inline Mat3 quat2rotm(Vec4 q){
    double qx=q[0],qy=q[1],qz=q[2],qw=q[3];
    return {{{1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw},
             {2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw},
             {2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy}}};
}

// rotation matrix → quaternion
inline Vec4 rotm2quat(Mat3 R){
    double tr=R[0][0]+R[1][1]+R[2][2];
    double qx,qy,qz,qw;
    if(tr>0){
        double S=std::sqrt(tr+1)*2;
        qw=0.25*S; qx=(R[2][1]-R[1][2])/S; qy=(R[0][2]-R[2][0])/S; qz=(R[1][0]-R[0][1])/S;
    } else if(R[0][0]>R[1][1] && R[0][0]>R[2][2]){
        double S=std::sqrt(1+R[0][0]-R[1][1]-R[2][2])*2;
        qw=(R[2][1]-R[1][2])/S; qx=0.25*S; qy=(R[0][1]+R[1][0])/S; qz=(R[0][2]+R[2][0])/S;
    } else if(R[1][1]>R[2][2]){
        double S=std::sqrt(1+R[1][1]-R[0][0]-R[2][2])*2;
        qw=(R[0][2]-R[2][0])/S; qx=(R[0][1]+R[1][0])/S; qy=0.25*S; qz=(R[1][2]+R[2][1])/S;
    } else {
        double S=std::sqrt(1+R[2][2]-R[0][0]-R[1][1])*2;
        qw=(R[1][0]-R[0][1])/S; qx=(R[0][2]+R[2][0])/S; qy=(R[1][2]+R[2][1])/S; qz=0.25*S;
    }
    return qnorm({qx,qy,qz,qw});
}

// Rodrigues exponential map: so(3) → SO(3)
inline Mat3 expm_so3(Vec3 phi){
    double theta=norm(phi);
    if(theta<1e-8){ return eye3()+skew(phi); }
    Vec3 ax=normalized(phi);
    Mat3 ax_sk=skew(ax);
    return eye3() + std::sin(theta)*ax_sk + (1-std::cos(theta))*(ax_sk*ax_sk);
}

// Rotation matrix → axis-angle vector
inline Vec3 rotm_to_axis_angle(Mat3 R){
    double tv=std::min(std::max((trace(R)-1)/2,-1.0),1.0);
    double theta=std::acos(tv);
    if(theta<1e-6) return {0,0,0};
    double s=theta/(2*std::sin(theta));
    return {s*(R[2][1]-R[1][2]), s*(R[0][2]-R[2][0]), s*(R[1][0]-R[0][1])};
}

// Quaternion multiply  q1*q2  [qx qy qz qw]
inline Vec4 quat_multiply(Vec4 q1, Vec4 q2){
    double w1=q1[3],x1=q1[0],y1=q1[1],z1=q1[2];
    double w2=q2[3],x2=q2[0],y2=q2[1],z2=q2[2];
    return {w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2,
            w1*z2+x1*y2-y1*x2+z1*w2,
            w1*w2-x1*x2-y1*y2-z1*z2};
}
inline Vec4 quat_inv(Vec4 q){return qnorm({-q[0],-q[1],-q[2],q[3]});}

// Angle between two rotation matrices (degrees)
inline double rot_angle_deg(Mat3 A, Mat3 B){
    Mat3 Rel=transpose(A)*B;
    double tv=std::min(std::max((trace(Rel)-1)/2,-1.0),1.0);
    double ang=std::acos(tv)*180.0/M_PI;
    return std::min(ang, 360.0-ang);
}

// ── 15×15 matrix helpers ──────────────────────────────────────────────────────
inline Mat15 eye15(){
    Mat15 M;
    for(int i=0;i<15;i++) M[i][i]=1.0;
    return M;
}
inline Mat15 mat15_add(const Mat15& A, const Mat15& B){
    Mat15 C;
    for(int i=0;i<15;i++) for(int j=0;j<15;j++) C[i][j]=A[i][j]+B[i][j];
    return C;
}
inline Mat15 mat15_scale(double s, const Mat15& A){
    Mat15 C;
    for(int i=0;i<15;i++) for(int j=0;j<15;j++) C[i][j]=s*A[i][j];
    return C;
}
inline Mat15 mat15_mul(const Mat15& A, const Mat15& B){
    Mat15 C;
    for(int i=0;i<15;i++) for(int k=0;k<15;k++) if(A[i][k]!=0)
        for(int j=0;j<15;j++) C[i][j]+=A[i][k]*B[k][j];
    return C;
}
inline Mat15 mat15_T(const Mat15& A){
    Mat15 T;
    for(int i=0;i<15;i++) for(int j=0;j<15;j++) T[i][j]=A[j][i];
    return T;
}
inline Vec15 mat15_vec(const Mat15& A, const Vec15& v){
    Vec15 r;
    for(int i=0;i<15;i++) for(int k=0;k<15;k++) r[i]+=A[i][k]*v[k];
    return r;
}
inline Mat15 mat15_symmetrize(Mat15 A, double eps=1e-8){
    for(int i=0;i<15;i++) for(int j=0;j<15;j++) A[i][j]=(A[i][j]+A[j][i])/2;
    for(int i=0;i<15;i++) A[i][i]+=eps;
    return A;
}

// ── 3×3 inversion (Cramer) ───────────────────────────────────────────────────
inline Mat3 inv3(Mat3 A){
    double det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                -A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                +A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if(std::abs(det)<1e-14) return eye3(); // fallback
    double id=1.0/det;
    return {{{ id*(A[1][1]*A[2][2]-A[1][2]*A[2][1]),
              -id*(A[0][1]*A[2][2]-A[0][2]*A[2][1]),
               id*(A[0][1]*A[1][2]-A[0][2]*A[1][1])},
             {-id*(A[1][0]*A[2][2]-A[1][2]*A[2][0]),
               id*(A[0][0]*A[2][2]-A[0][2]*A[2][0]),
              -id*(A[0][0]*A[1][2]-A[0][2]*A[1][0])},
             { id*(A[1][0]*A[2][1]-A[1][1]*A[2][0]),
              -id*(A[0][0]*A[2][1]-A[0][1]*A[2][0]),
               id*(A[0][0]*A[1][1]-A[0][1]*A[1][0])}}};
}

// ── Helpers: P*H'  (15×15 · 15×3 → 15×3) ────────────────────────────────────
// H is stored as [rows=3][cols=15] (row-major measurement matrix)
using H3  = std::array<std::array<double,15>,3>;  // 3×15
using PH3 = std::array<std::array<double,3>,15>;  // 15×3

inline PH3 mat_PH(const Mat15& P, const H3& H){
    PH3 R{};
    for(int i=0;i<15;i++) for(int j=0;j<3;j++)
        for(int k=0;k<15;k++) R[i][j]+=P[i][k]*H[j][k];
    return R;
}
// H·P·H'  →  3×3
inline Mat3 mat_HPH(const H3& H, const Mat15& P){
    PH3 ph=mat_PH(P,H);
    Mat3 S{};
    for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        for(int k=0;k<15;k++) S[i][j]+=H[i][k]*ph[k][j];
    return S;
}
// K = PH' · S^{-1}  (15×3)
inline PH3 mat_Kgain(const PH3& PH, Mat3 Sinv){
    PH3 K{};
    for(int i=0;i<15;i++) for(int j=0;j<3;j++)
        for(int k=0;k<3;k++) K[i][j]+=PH[i][k]*Sinv[k][j];
    return K;
}
// delta_x = K * innov  (15×1)
inline Vec15 mat_Kinnov(const PH3& K, Vec3 z){
    Vec15 dx;
    for(int i=0;i<15;i++) dx[i]=K[i][0]*z[0]+K[i][1]*z[1]+K[i][2]*z[2];
    return dx;
}
// Joseph form:  (I-K*H)*P*(I-K*H)' + K*R*K'
inline Mat15 joseph_update(const Mat15& P, const PH3& K, const H3& H, double sigma2){
    // I - K*H
    Mat15 IKH=eye15();
    for(int i=0;i<15;i++) for(int j=0;j<15;j++)
        for(int k=0;k<3;k++) IKH[i][j]-=K[i][k]*H[k][j];
    // (I-KH)*P*(I-KH)'
    Mat15 tmp=mat15_mul(IKH,P);
    Mat15 Pn=mat15_mul(tmp,mat15_T(IKH));
    // + K*R*K'   (R = sigma2*I3)
    for(int i=0;i<15;i++) for(int j=0;j<15;j++)
        for(int k=0;k<3;k++) Pn[i][j]+=sigma2*K[i][k]*K[j][k];
    return mat15_symmetrize(Pn);
}

// ── 6×6 innovation for combined pos+ori update ───────────────────────────────
using H6  = std::array<std::array<double,15>,6>;
using PH6 = std::array<std::array<double,6>,15>;
struct Mat6{ double m[6][6]{}; };

inline PH6 mat_PH6(const Mat15& P, const H6& H){
    PH6 R{};
    for(int i=0;i<15;i++) for(int j=0;j<6;j++)
        for(int k=0;k<15;k++) R[i][j]+=P[i][k]*H[j][k];
    return R;
}
inline Mat6 mat_HPH6(const H6& H, const Mat15& P){
    PH6 ph=mat_PH6(P,H);
    Mat6 S;
    for(int i=0;i<6;i++) for(int j=0;j<6;j++)
        for(int k=0;k<15;k++) S.m[i][j]+=H[i][k]*ph[k][j];
    return S;
}
// Simple 6×6 inversion via block diagonal (R is block-diag(R_ori, R_pos))
inline Mat6 inv6_blockdiag(Mat6 S){
    // Use cofactor / LU for 6×6
    // For block-diagonal structure this is fine as 2× 3×3 inversions
    // But S won't generally be block-diagonal after adding P terms
    // → use full LU
    // Simple Gauss-Jordan
    double a[6][12]{};
    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++) a[i][j]=S.m[i][j];
        a[i][6+i]=1.0;
    }
    for(int col=0;col<6;col++){
        int pivot=col;
        for(int row=col+1;row<6;row++) if(std::abs(a[row][col])>std::abs(a[pivot][col])) pivot=row;
        std::swap(a[col],a[pivot]);
        double sc=a[col][col]; if(std::abs(sc)<1e-14) sc=1e-14;
        for(int j=0;j<12;j++) a[col][j]/=sc;
        for(int row=0;row<6;row++) if(row!=col){
            double f=a[row][col];
            for(int j=0;j<12;j++) a[row][j]-=f*a[col][j];
        }
    }
    Mat6 inv;
    for(int i=0;i<6;i++) for(int j=0;j<6;j++) inv.m[i][j]=a[i][6+j];
    return inv;
}
inline std::array<double,6> mat6_vec(const Mat6& A, const std::array<double,6>& v){
    std::array<double,6> r{};
    for(int i=0;i<6;i++) for(int j=0;j<6;j++) r[i]+=A.m[i][j]*v[j];
    return r;
}
inline PH6 mat_Kgain6(const PH6& PH, const Mat6& Sinv){
    PH6 K{};
    for(int i=0;i<15;i++) for(int j=0;j<6;j++)
        for(int k=0;k<6;k++) K[i][j]+=PH[i][k]*Sinv.m[k][j];
    return K;
}
inline Vec15 mat_Kinnov6(const PH6& K, const std::array<double,6>& z){
    Vec15 dx;
    for(int i=0;i<15;i++) for(int j=0;j<6;j++) dx[i]+=K[i][j]*z[j];
    return dx;
}
inline Mat15 joseph_update6(const Mat15& P, const PH6& K, const H6& H,
                             double sigma_ori2, double sigma_pos2){
    Mat15 IKH=eye15();
    for(int i=0;i<15;i++) for(int j=0;j<15;j++)
        for(int k=0;k<6;k++) IKH[i][j]-=K[i][k]*H[k][j];
    Mat15 tmp=mat15_mul(IKH,P);
    Mat15 Pn=mat15_mul(tmp,mat15_T(IKH));
    // K*R*K'  R=diag(s_ori2*I3, s_pos2*I3)
    for(int i=0;i<15;i++) for(int j=0;j<15;j++){
        double s=0;
        for(int k=0;k<3;k++) s+=sigma_ori2*K[i][k]*K[j][k];
        for(int k=3;k<6;k++) s+=sigma_pos2*K[i][k]*K[j][k];
        Pn[i][j]+=s;
    }
    return mat15_symmetrize(Pn);
}
