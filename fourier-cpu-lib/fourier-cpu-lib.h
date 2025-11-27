#pragma once

#include <vector>
#include <string>

// Define export macro based on project preprocessor definitions
#ifdef FOURIERCPULIB_EXPORTS
#define CPU_API __declspec(dllexport)
#else
#define CPU_API __declspec(dllimport)
#endif

namespace FourierCPU {
    // Parameters received from UI
    struct Params {
        int numHarmonics;
    };

// Parameters returned to UI 
struct Result {
    std::vector<double> calculatedY;   // calculated series points (Yg)
    std::vector<double> c_amplitudes;  // harmonic amplitudes (c)
    std::vector<double> a_coeffs;      // coeffs (a)
    std::vector<double> b_coeffs;      // coeffs (b)
    float executionTimeMs;             // telemetry
    bool isSuccess;                    // success flag
    std::string errorMessage;          // error details
};

class CPU_API FourierCpuCalculator {
public:
    FourierCpuCalculator();
    ~FourierCpuCalculator();

    // Device selection methods removed

    Result Calculate(
        const Params& params,
        const std::vector<double>& x_values,
        const std::vector<double>& y_values
    );
};
};