// =============================================================================
// benchmark_metrics.hpp  –  Comprehensive computational performance metrics
//                           CPU utilization, memory, latency distributions,
//                           real-time factor, jitter, update frequency
// =============================================================================
#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <sstream>

#ifdef _WIN32
  #include <windows.h>
  #include <psapi.h>
#else
  #include <sys/resource.h>
  #include <unistd.h>
#endif

// ── Per-step timing record ────────────────────────────────────────────────────
struct StepRecord {
    double wall_ms;     // wall-clock time for this step
    double cpu_ms;      // CPU time for this step (if available)
};

// ── Full benchmark metrics for one filter ────────────────────────────────────
struct BenchmarkMetrics {
    std::string filter_name;
    std::string sequence;

    // Timing statistics
    double mean_ms      {};
    double median_ms    {};
    double std_ms       {};
    double min_ms       {};
    double max_ms       {};
    double p95_ms       {};
    double p99_ms       {};
    double p999_ms      {};

    // Real-time metrics
    double budget_hz        {200.0};    // target frequency
    double budget_ms        {5.0};      // = 1000/budget_hz
    double rt_factor        {};         // budget_ms / mean_ms  (>1 = real-time capable)
    double rt_margin_pct    {};         // (1 - mean/budget)*100
    double budget_violations{};         // % of steps exceeding budget
    double max_freq_hz      {};         // 1000/mean_ms

    // Jitter
    double jitter_ms    {};             // std of inter-step variation
    double cv_pct       {};             // coefficient of variation (std/mean)*100

    // Memory
    double peak_mem_mb  {};             // peak RSS in MB
    double delta_mem_mb {};             // memory used during run

    // CPU
    double cpu_util_pct {};             // estimated CPU utilization %

    // Accuracy
    double pos_rmse_m   {};
    double ori_rmse_deg {};

    // Total
    int    n_steps      {};
    double total_wall_s {};
};

// ── Memory query helpers ─────────────────────────────────────────────────────
inline double get_rss_mb(){
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if(GetProcessMemoryInfo(GetCurrentProcess(),&pmc,sizeof(pmc)))
        return (double)pmc.WorkingSetSize / (1024.0*1024.0);
    return 0.0;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF,&usage);
    return (double)usage.ru_maxrss / 1024.0;   // Linux: kB → MB
#endif
}

inline double get_cpu_time_ms(){
#ifdef _WIN32
    FILETIME ct,et,kt,ut;
    if(GetProcessTimes(GetCurrentProcess(),&ct,&et,&kt,&ut)){
        ULARGE_INTEGER u; u.LowPart=ut.dwLowDateTime; u.HighPart=ut.dwHighDateTime;
        return (double)u.QuadPart / 10000.0;   // 100ns → ms
    }
    return 0.0;
#else
    struct rusage usage; getrusage(RUSAGE_SELF,&usage);
    return usage.ru_utime.tv_sec*1000.0 + usage.ru_utime.tv_usec/1000.0;
#endif
}

// ── Percentile helper ─────────────────────────────────────────────────────────
inline double percentile(std::vector<double> v, double pct){
    if(v.empty()) return 0.0;
    std::sort(v.begin(),v.end());
    double idx=(pct/100.0)*(v.size()-1);
    int lo=(int)idx; int hi=lo+1;
    if(hi>=(int)v.size()) return v.back();
    double frac=idx-lo;
    return v[lo]*(1-frac)+v[hi]*frac;
}

// ── Compute all metrics from timing vector ────────────────────────────────────
inline BenchmarkMetrics compute_metrics(
    const std::string& name,
    const std::string& seq,
    const std::vector<double>& timing_ms_all,
    double pos_rmse, double ori_rmse,
    double mem_before_mb, double cpu_before_ms)
{
    BenchmarkMetrics m;
    m.filter_name = name;
    m.sequence    = seq;
    m.pos_rmse_m  = pos_rmse;
    m.ori_rmse_deg= ori_rmse;

    // Filter out zeros and extreme outliers (>50ms = OS preemption)
    std::vector<double> v;
    for(double t: timing_ms_all) if(t>0.0 && t<50.0) v.push_back(t);

    m.n_steps = (int)v.size();
    if(v.empty()){ return m; }

    // Basic stats
    double sum=std::accumulate(v.begin(),v.end(),0.0);
    m.mean_ms  = sum/v.size();
    m.median_ms= percentile(v,50);
    m.min_ms   = *std::min_element(v.begin(),v.end());
    m.max_ms   = *std::max_element(v.begin(),v.end());
    m.p95_ms   = percentile(v,95);
    m.p99_ms   = percentile(v,99);
    m.p999_ms  = percentile(v,99.9);

    // Std dev
    double sq=0; for(double t:v) sq+=(t-m.mean_ms)*(t-m.mean_ms);
    m.std_ms = std::sqrt(sq/v.size());
    m.cv_pct = (m.mean_ms>0)? (m.std_ms/m.mean_ms)*100 : 0;

    // Jitter (consecutive differences)
    std::vector<double> diffs;
    for(int i=1;i<(int)v.size();i++) diffs.push_back(std::abs(v[i]-v[i-1]));
    if(!diffs.empty()){
        double dsum=std::accumulate(diffs.begin(),diffs.end(),0.0);
        double dmean=dsum/diffs.size();
        double dsq=0; for(double d:diffs) dsq+=(d-dmean)*(d-dmean);
        m.jitter_ms=std::sqrt(dsq/diffs.size());
    }

    // Real-time metrics
    m.budget_ms     = 1000.0/m.budget_hz;
    m.rt_factor     = m.budget_ms/m.mean_ms;
    m.rt_margin_pct = (1.0-m.mean_ms/m.budget_ms)*100.0;
    m.max_freq_hz   = (m.mean_ms>0)? 1000.0/m.mean_ms : 0;

    // Budget violations
    int viol=0; for(double t:v) if(t>m.budget_ms) viol++;
    m.budget_violations=(double)viol/v.size()*100.0;

    // Total wall time
    double sum_all=0; for(double t:timing_ms_all) sum_all+=t;
    m.total_wall_s=sum_all/1000.0;

    // Memory
    m.peak_mem_mb = get_rss_mb();
    m.delta_mem_mb= m.peak_mem_mb - mem_before_mb;

    // CPU utilization estimate
    double cpu_now=get_cpu_time_ms();
    double cpu_used=cpu_now-cpu_before_ms;
    if(m.total_wall_s>0)
        m.cpu_util_pct=cpu_used/(m.total_wall_s*1000.0)*100.0;

    return m;
}

// ── Print detailed metrics report ─────────────────────────────────────────────
// In benchmark_metrics.hpp - replace the box-drawing function with this:

inline void print_metrics(const BenchmarkMetrics& m) {
    printf("\n");
    printf("========================================\n");
    printf(" BENCHMARK METRICS: %s [%s]\n", m.filter_name.c_str(), m.sequence.c_str());
    printf("========================================\n");

    printf("\n--- TIMING STATISTICS ---\n");
    printf("  Steps analyzed   :    %d\n",      m.n_steps);
    printf("  Mean             :   %.4f ms\n",  m.mean_ms);
    printf("  Median           :   %.4f ms\n",  m.median_ms);
    printf("  Std Dev (jitter) :   %.4f ms  (CV=%.1f%%)\n", m.std_ms, m.cv_pct);
    printf("  Min              :   %.4f ms\n",  m.min_ms);
    printf("  Max (worst-case) :   %.4f ms\n",  m.max_ms);
    printf("  P95              :   %.4f ms\n",  m.p95_ms);
    printf("  P99              :   %.4f ms\n",  m.p99_ms);
    printf("  P99.9            :   %.4f ms\n",  m.p999_ms);

    printf("\n--- REAL-TIME PERFORMANCE ---\n");
    printf("  Target frequency :   %.1f Hz\n",  m.budget_hz);
    printf("  Max frequency    :   %.1f Hz\n",  m.max_freq_hz);
    printf("  RT factor        :   %.1fx  (>1 = real-time)\n", m.rt_factor);
    printf("  RT margin        :   %.1f%%\n",   m.rt_margin_pct);
    printf("  Budget violations:   %.2f%% of steps\n", m.budget_violations);
    printf("  Jitter           :   %.4f ms\n",  m.jitter_ms);

    printf("\n--- RESOURCE USAGE ---\n");
    printf("  Peak memory      :   %.2f MB\n",  m.peak_mem_mb);
    printf("  Memory delta     :   %.2f MB\n",  m.delta_mem_mb);
    printf("  CPU utilization  :   %.1f%%\n",   m.cpu_util_pct);
    printf("  Total wall time  :   %.3f s\n",   m.total_wall_s);

    printf("\n--- ACCURACY ---\n");
    printf("  Position RMSE    :   %.4f m\n",   m.pos_rmse_m);
    printf("  Orientation RMSE :   %.3f deg\n", m.ori_rmse_deg);
    printf("========================================\n");
}
// ── Save metrics to CSV ───────────────────────────────────────────────────────
inline void save_metrics_csv(const std::vector<BenchmarkMetrics>& all,
                              const std::string& fname){
    std::ofstream f(fname);
    f<<"filter,sequence,n_steps,mean_ms,median_ms,std_ms,min_ms,max_ms,"
      "p95_ms,p99_ms,p999_ms,jitter_ms,cv_pct,"
      "max_freq_hz,rt_factor,rt_margin_pct,budget_violations_pct,"
      "peak_mem_mb,delta_mem_mb,cpu_util_pct,total_wall_s,"
      "pos_rmse_m,ori_rmse_deg\n";
    for(auto& m:all)
        f<<m.filter_name<<','<<m.sequence<<','<<m.n_steps<<','
         <<m.mean_ms<<','<<m.median_ms<<','<<m.std_ms<<','
         <<m.min_ms<<','<<m.max_ms<<','
         <<m.p95_ms<<','<<m.p99_ms<<','<<m.p999_ms<<','
         <<m.jitter_ms<<','<<m.cv_pct<<','
         <<m.max_freq_hz<<','<<m.rt_factor<<','
         <<m.rt_margin_pct<<','<<m.budget_violations<<','
         <<m.peak_mem_mb<<','<<m.delta_mem_mb<<','
         <<m.cpu_util_pct<<','<<m.total_wall_s<<','
         <<m.pos_rmse_m<<','<<m.ori_rmse_deg<<'\n';
    printf("  Metrics saved: %s\n",fname.c_str());
}

// ── Print comparison table ────────────────────────────────────────────────────
inline void print_comparison_table(const std::vector<BenchmarkMetrics>& all){
    printf("\n");
    printf("=============================================================================================================================\n");
    printf("  COMPREHENSIVE COMPUTATIONAL BENCHMARK RESULTS\n");
    printf("=============================================================================================================================\n");
    printf("  %-12s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-8s | %-8s\n",
           "Filter","Mean(ms)","Std(ms)","P99(ms)","Max(ms)","MaxFreq","RTFactor","Budget%","Memory");
    printf("  -----------------------------------------------------------------------------------------------------------------------------\n");
    for(auto& m:all)
        printf("  %-12s | %9.4f | %9.4f | %9.4f | %9.4f | %7.0fHz | %7.1fx  | %6.1f%%  | %6.1fMB\n",
               m.filter_name.c_str(),
               m.mean_ms, m.std_ms, m.p99_ms, m.max_ms,
               m.max_freq_hz, m.rt_factor,
               m.rt_margin_pct, m.peak_mem_mb);
    printf("=============================================================================================================================\n");

    double ukf_mean=0, ukf_p99=0;
    for(auto& m:all) if(m.filter_name=="Full_UKF"){ ukf_mean=m.mean_ms; ukf_p99=m.p99_ms; }
    if(ukf_mean>0){
        printf("\n  Speedup vs Full UKF:\n");
        for(auto& m:all) if(m.filter_name!="Full_UKF")
            printf("  %-12s : %.1fx faster (mean)  |  %.1fx better worst-case (P99)\n",
                   m.filter_name.c_str(), ukf_mean/m.mean_ms, ukf_p99/m.p99_ms);
    }
    printf("\n");
}
