#pragma once
// =============================================================================
// vio_params.hpp – Unified noise parameters for fair comparison
// =============================================================================
#define _USE_MATH_DEFINES
#include <cmath>

namespace VIOParams {
    // ===== IMU Noise (Typical MEMS, e.g., BMI088) =====
    constexpr double GYRO_NOISE_SIGMA = 0.0002;      // rad/s/√Hz (200 μrad/s)
    constexpr double ACCEL_NOISE_SIGMA = 0.002;      // m/s²/√Hz (2 mg)
    
    // Bias random walk (Allen deviation)
    constexpr double GYRO_BIAS_WALK = 0.00001;       // rad/s²/√Hz
    constexpr double ACCEL_BIAS_WALK = 0.0001;       // m/s³/√Hz
    
    // Bias correlation times (seconds)
    constexpr double TAU_GYRO = 100.0;               // 100 seconds
    constexpr double TAU_ACCEL = 100.0;              // 100 seconds
    
    // ===== Visual Odometry Noise (Good VO, e.g., OKVIS) =====
    constexpr double VO_POS_SIGMA = 0.05;            // meters (NOT 0.15!)
    constexpr double VO_ORI_SIGMA_DEG = 0.5;         // degrees
    constexpr double VO_ORI_SIGMA_RAD = VO_ORI_SIGMA_DEG * M_PI / 180.0;  // 0.00873 rad
    
    // ===== ZUPT Parameters =====
    constexpr double ZUPT_VEL_SIGMA = 0.01;           // m/s
    constexpr double ZUPT_OMEGA_THRESH = 0.01;        // rad/s
    constexpr double ZUPT_ACC_THRESH = 0.05;          // m/s² (|norm - 9.81|)
    constexpr double ZUPT_GAIN_SCALE = 1.0;           // NO scaling! (was 0.3)
    
    // ===== Gravity Alignment =====
    constexpr double GRAVITY_ALIGN_SIGMA = 0.05;      // rad (~2.86°)
    constexpr double GRAVITY_ACC_THRESH = 0.1;        // m/s²
    constexpr double GRAVITY_GAIN_SCALE = 1.0;        // NO scaling! (was 0.5)
    
    // ===== Initial Covariances =====
    constexpr double INIT_ORI_SIGMA_DEG = 0.5;        // degrees
    constexpr double INIT_ORI_SIGMA_RAD = INIT_ORI_SIGMA_DEG * M_PI / 180.0;
    constexpr double INIT_POS_SIGMA = 0.01;           // meters
    constexpr double INIT_VEL_SIGMA = 0.05;           // m/s
    constexpr double INIT_GYRO_BIAS_SIGMA = 0.001;    // rad/s
    constexpr double INIT_ACCEL_BIAS_SIGMA = 0.01;    // m/s²
    
    // ===== Process Noise for ESKF (continuous-time) =====
    inline double gyro_noise_power() { return GYRO_NOISE_SIGMA * GYRO_NOISE_SIGMA; }
    inline double accel_noise_power() { return ACCEL_NOISE_SIGMA * ACCEL_NOISE_SIGMA; }
    inline double gyro_bias_power() { return 2.0 * GYRO_BIAS_WALK * GYRO_BIAS_WALK / TAU_GYRO; }
    inline double accel_bias_power() { return 2.0 * ACCEL_BIAS_WALK * ACCEL_BIAS_WALK / TAU_ACCEL; }
}