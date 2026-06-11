// rng.h
#pragma once
#include <random>
#include <array>
#include "math_utils.hpp"

// Simple RNG struct - complete definition, no forward declaration needed
struct RNG {
    std::mt19937_64 eng;
    std::normal_distribution<double> nd;
    
    RNG() : eng(42), nd(0.0, 1.0) {}
    RNG(uint64_t seed) : eng(seed), nd(0.0, 1.0) {}
    
    double randn() { return nd(eng); }
    
    Vec3 randn3() { 
        return {nd(eng), nd(eng), nd(eng)}; 
    }
};