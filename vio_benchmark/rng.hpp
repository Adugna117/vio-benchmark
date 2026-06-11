// rng.hpp
#pragma once
#include <random>
#include <array>
#include "math_utils.hpp"

// Unified Random Number Generator for all filters
struct RNG {
    std::mt19937_64 eng;
    std::normal_distribution<double> nd{0.0, 1.0};
    
    // Default constructor with fixed seed (for reproducibility)
    RNG() : eng(42) {}
    
    // Constructor with explicit seed (for Monte Carlo)
    RNG(uint64_t seed) : eng(seed) {}
    
    // Copy constructor
    RNG(const RNG& other) : eng(other.eng) {}
    
    // Assignment operator
    RNG& operator=(const RNG& other) {
        if (this != &other) {
            eng = other.eng;
        }
        return *this;
    }
    
    // Reseed the generator
    void seed(uint64_t s) { eng.seed(s); }
    
    // Generate a single Gaussian sample N(0,1)
    double randn() { return nd(eng); }
    
    // Generate 3 independent Gaussian samples
    Vec3 randn3() { 
        return {nd(eng), nd(eng), nd(eng)}; 
    }
    
    // Generate 6 independent Gaussian samples (for combined update)
    std::array<double, 6> randn6() {
        return {nd(eng), nd(eng), nd(eng), nd(eng), nd(eng), nd(eng)};
    }
};