#include "fourier-cpu-lib.h"
#include <cmath>
#include <chrono> 
#include <numeric>  
#include <stdexcept>

const double PI = 3.14159265358979323846;

using namespace FourierCPU;

FourierCpuCalculator::FourierCpuCalculator() {}

FourierCpuCalculator::~FourierCpuCalculator() {}

Result FourierCpuCalculator::Calculate(const Params& params, const std::vector<double>& x_values, const std::vector<double>& y_values) {
    Result result;

    // time start
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        if (x_values.size() != y_values.size() || x_values.empty()) {
            throw std::runtime_error("Bad input vectors");
        }

        const int Ne = (int)x_values.size();
        const int Ng = params.numHarmonics;
        const double Tp = x_values.back() - x_values.front();
        if (Tp <= 0) throw std::runtime_error("Bad period (bad a-b range)");

        const double w = 2.0 * PI / Tp;

        // size output vectors
        result.a_coeffs.resize(Ng + 1);
        result.b_coeffs.resize(Ng + 1);
        result.c_amplitudes.resize(Ng + 1);
        result.calculatedY.resize(Ne);

        // calculating a0
        double sum_y = std::accumulate(y_values.begin(), y_values.end(), 0.0);
        double a0 = sum_y / Ne;

        result.a_coeffs[0] = a0;
        result.b_coeffs[0] = 0.0;
        result.c_amplitudes[0] = 0.0;

        // calculating coeffs
        for (int k = 1; k <= Ng; ++k) {
            double G_sum = 0.0;
            double D_sum = 0.0;

            for (int i = 0; i < Ne; ++i) {
                double S = k * w * x_values[i];
                G_sum += y_values[i] * cos(S);
                D_sum += y_values[i] * sin(S);
            }

            // normalize coeffs
            result.a_coeffs[k] = G_sum * 2.0 / Ne;
            result.b_coeffs[k] = D_sum * 2.0 / Ne;

            // calculate amplitude
            result.c_amplitudes[k] = std::sqrt(pow(result.a_coeffs[k], 2) + pow(result.b_coeffs[k], 2));
        }

        // reconstruct series - iterate through data and add to harmonics
        for (int i = 0; i < Ne; ++i) {
            double D_val = x_values[i] * w;
            double S = 0.0;

            for (int k = 1; k <= Ng; ++k) {
                double KOM = k * D_val;
                S += result.b_coeffs[k] * sin(KOM) + result.a_coeffs[k] * cos(KOM);
            }

            result.calculatedY[i] = a0 + S;
        }

        result.isSuccess = true;
    }
    catch (const std::exception& e) {
        result.isSuccess = false;
        result.errorMessage = e.what();
        result.calculatedY.clear();
        result.a_coeffs.clear();
        result.b_coeffs.clear();
        result.c_amplitudes.clear();
    }

    
    auto end_time = std::chrono::high_resolution_clock::now(); // time stop
    
    std::chrono::duration<float, std::milli> duration = end_time - start_time;// convert time to ms
    result.executionTimeMs = duration.count();

    return result;
}