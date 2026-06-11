# VIO Filter Benchmark – C++17

Pure C++17 implementation of three VIO filtering approaches for the paper:
*"Accuracy–Computational Trade-offs of VIO Filtering on Consumer UAV Hardware"*

## Filters implemented
| Filter | Description |
|--------|-------------|
| **ESKF** | Error-State Kalman Filter (15 states) |
| **Full UKF** | Full 15-state Unscented Kalman Filter with 31 sigma points |
| **Hybrid** | ESKF for position/velocity/bias + 4-state quaternion UKF for orientation |

All filters include ZUPT (Zero-velocity Update) and gravity-alignment pseudo-measurements.

---

## Build – MSYS2 UCRT64

### Prerequisites (install once)
```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
pacman -S mingw-w64-ucrt-x86_64-cmake   # optional, for CMake build
```

### Option A – direct g++ (fastest)
```bash
cd vio_benchmark
g++ -O3 -std=c++17 -march=native -ffast-math -funroll-loops \
    -o vio_benchmark.exe main.cpp
```

### Option B – CMake
```bash
mkdir build && cd build
cmake .. -G "MSYS Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Run

### Synthetic demo (no data files needed)
```bash
./vio_benchmark.exe --synthetic
```

### EuRoC MAV dataset
Download from https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf (EuRoC website).

Expected CSV format:
- **IMU**: `timestamp[ns], omega_x, omega_y, omega_z, a_x, a_y, a_z`
- **GT**:  `timestamp[ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, ...`

```bash
./vio_benchmark.exe  imu0/data.csv  state_groundtruth_estimate0/data.csv  MH01
```

For each EuRoC sequence (MH01–MH04):
```bash
for seq in MH01 MH02 MH03 MH04; do
    ./vio_benchmark.exe \
        /path/to/${seq}/mav0/imu0/data.csv \
        /path/to/${seq}/mav0/state_groundtruth_estimate0/data.csv \
        ${seq}
done
```

---

## Output files

| File | Contents |
|------|----------|
| `<seq>_eskf.csv` | Per-step estimated pose + ground truth + timing |
| `<seq>_full_ukf.csv` | Same for Full UKF |
| `<seq>_hybrid.csv` | Same for Hybrid |
| `<seq>_summary.csv` | RMSE + avg time for all three filters |

### Summary CSV columns
```
filter, pos_rmse_m, ori_rmse_deg, avg_time_ms
```

---

## Project structure

```
vio_benchmark/
├── main.cpp            – benchmark driver + synthetic data generator
├── math_utils.hpp      – Eigen-free matrix/quaternion/SO3 math
├── data_io.hpp         – EuRoC CSV loader, data alignment, RMSE
├── eskf.hpp            – ESKF (15-state)
├── full_ukf.hpp        – Full UKF (15-state, 31 sigma points)
├── hybrid_filter.hpp   – Hybrid ESKF + 4-state quaternion UKF
└── CMakeLists.txt      – CMake build file
```

---

## Expected results (paper Table I)

| Filter   | Pos RMSE (m) | Ori RMSE (°) | Avg time (ms/step) |
|----------|--------------|--------------|--------------------|
| ESKF     | ~0.15–0.25   | ~1.7         | ~0.0008            |
| Full UKF | ~0.10–0.15   | ~2.0–2.5     | ~0.22              |
| Hybrid   | ~0.60–0.70   | ~2.3         | ~0.01              |

*Exact numbers depend on sequence, hardware, and compiler.*

---

## Notes on the implementation

- **No external dependencies** – no Eigen, no BLAS.  All linear algebra uses
  fixed-size stack arrays for cache efficiency.
- **Cholesky decomposition** implemented in-place for 4×4 (QUKF) and 15×15
  (Full UKF sigma points).
- **Camera measurements** are simulated from interpolated ground truth with
  additive Gaussian noise (`σ_pos = 0.15 m`, `σ_ori = 0.5°`), matching the
  MATLAB reference implementation.
- The **QUKF** inside the Hybrid filter uses 9 sigma points (n=4), drastically
  reducing cost vs. the 31-point Full UKF.
- High-resolution timing uses `std::chrono::high_resolution_clock` per filter
  step; wall-clock outliers >10 ms are excluded from the average (OS scheduling
  jitter).
