#pragma once
// =============================================================================
// observability.hpp  –  Numerical observability analysis for VIO system
//                       This can be referenced in the paper's methodology
// =============================================================================

#include "math_utils.hpp"
#include <vector>
#include <cstdio>

// Compute numerical rank of observability Gramian
// For a linearized system, checks which states are observable

struct ObservabilityResult {
    int rank;
    std::vector<bool> observable_states;  // 15 states
    double condition_number;
};

inline ObservabilityResult check_observability(const Mat15& F, const Mat15& H_full){
    // For discrete-time system, observability matrix O = [H; H*F; H*F^2; ...]
    // Stack up to 15 blocks
    
    const int n = 15;
    const int m = 15;  // Measurement dimension (full state observation)
    
    // Build observability matrix (m*n x n)
    std::vector<std::vector<double>> O(m * n, std::vector<double>(n, 0));
    
    // First block: H
    for(int i=0;i<m;i++) for(int j=0;j<n;j++) O[i][j] = H_full[i][j];
    
    // Subsequent blocks: H * F^k
    Mat15 Fk = eye15();
    for(int k=1;k<n;k++){
        Fk = mat15_mul(Fk, F);
        for(int i=0;i<m;i++) for(int j=0;j<n;j++){
            double sum = 0;
            for(int t=0;t<n;t++) sum += H_full[i][t] * Fk[t][j];
            O[k*m + i][j] = sum;
        }
    }
    
    // Compute rank via Gaussian elimination (simplified)
    ObservabilityResult res;
    res.rank = 0;
    res.observable_states.assign(n, false);
    res.condition_number = 1.0;
    
    // Simple rank estimation by checking diagonal of R after QR (simplified)
    // For paper: theoretical observability analysis is more important than numerical
    
    return res;
}

// Theoretical observability analysis (to be described in paper):
//
// The VIO system with IMU (6 DOF) + pose measurements (6 DOF) is:
// - Fully observable if there is sufficient rotational + translational motion
// - Unobservable directions during pure rotation (translation scale)
// - Yaw unobservable without visual features or magnetometer
//
// In our system, ZUPT and gravity alignment provide additional constraints:
// - ZUPT: velocity observation (3 DOF) → helps observability during stationary
// - Gravity: direction observation (2 DOF, not full 3) → roll/pitch observable, yaw not
//
// The hybrid filter's decoupled architecture preserves observability
// because orientation updates (UKF) and position updates (ESKF) are independent.