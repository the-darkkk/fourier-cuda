#include <cmath>
#include "fourier-lib.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <fstream>  // for logging
#include <string> // for logging

using namespace FourierGPU;

__global__ void calculateCoefficientsKernel(double*, double*, const double*, const double*, int, int, double);
__global__ void calculateSeriesKernel(double*, const double*, const double*, const double*, int, int, double);
__global__ void normalizeCoefficientsKernel(double*, double*, const double*, const double*, int, int);

void LogGpuError(const std::string& context) { // logging function (shouldn't bother you if everything is good)
    cudaError_t err = cudaGetLastError();
    cudaError_t syncErr = cudaDeviceSynchronize();

    if (err != cudaSuccess || syncErr != cudaSuccess) {
        std::ofstream outfile("cuda_log.txt", std::ios_base::app); // Append mode
        if (err != cudaSuccess) {
            outfile << "LAUNCH ERROR [" << context << "]: " << cudaGetErrorString(err) << "\n";
        }
        if (syncErr != cudaSuccess) {
            outfile << "EXECUTION ERROR [" << context << "]: " << cudaGetErrorString(syncErr) << "\n";
        }
        outfile.close();
    }
}

FourierCudaCalculator::FourierCudaCalculator() {
	selectedDeviceIndex = -1; // initializing the selected device index
};

FourierCudaCalculator::~FourierCudaCalculator() {};

std::vector<std::string> FourierCudaCalculator::GetAvailableDevices() {
	std::vector<std::string> devices;
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	
	if (error_id != cudaSuccess) {
		return devices;
	}
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        devices.push_back(std::string(props.name));
    }
    
	return devices;
};

bool FourierCudaCalculator::SelectDevice(int deviceId) {
    cudaError_t error_id = cudaSetDevice(deviceId);

    if (error_id == cudaSuccess) {
        this->selectedDeviceIndex = deviceId;
        return true;
    }
    else {
        this->selectedDeviceIndex = -1;
        return false;
    }
};

Result FourierCudaCalculator::Calculate(const Params& params, const std::vector<double>& x_values, const std::vector<double>& y_values) {
    std::ofstream("cuda_log.txt", std::ios::trunc);
    Result result;
    const double PI = 3.14159265358979323846;   // a kludge bcs M_PI somewhy does not work here
    if (selectedDeviceIndex < 0) {
        result.isSuccess = false;
        result.errorMessage = "GPU device is not selected.";
        return result;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    try {
        const int Ne = x_values.size();
        const int Ng = params.numHarmonics;
        const double Tp = x_values.back() - x_values.front();   // assuming that period is the given range of digits
        const double w = 2.0 * PI / Tp;     // calculating the edge frequency

        double a0_sum = 0.0; // calculate a0
        for (const auto& y : y_values) a0_sum += y;
        const double a0 = a0_sum / Ne;

        // allocate the vram for calculation
        double* d_x, * d_y, * d_G, * d_D, * d_a, * d_b, * d_Yg;
        const size_t size_Ne = Ne * sizeof(double);     // allocate the size needed for the input dots array
        const size_t size_Ng = (Ng + 1) * sizeof(double);   // allocate the size needed for the output harmonics array
        cudaMalloc(&d_x, size_Ne);
        cudaMalloc(&d_y, size_Ne);
        cudaMalloc(&d_G, size_Ng); 
        cudaMalloc(&d_D, size_Ng);
        cudaMalloc(&d_a, size_Ng); 
        cudaMalloc(&d_b, size_Ng);
        cudaMalloc(&d_Yg, size_Ne); 

        cudaEventRecord(start); // start time measurement - needed for time telemetry

        // move the data from ram to vram
        cudaMemcpy(d_x, x_values.data(), size_Ne, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y_values.data(), size_Ne, cudaMemcpyHostToDevice);
        cudaMemset(d_G, 0, size_Ng); // zeroing the memory - i guess it's needed
        cudaMemset(d_D, 0, size_Ng);

        const int threadsPerBlock = 256; // defining the constant

        calculateCoefficientsKernel << <Ng, threadsPerBlock >> > (d_G, d_D, d_x, d_y, Ne, Ng, w);   // launching a parallel fourier coefficients calculation
        LogGpuError("calculateCoefficientsKernel");
        
        const int blocksForNorm = (Ng + threadsPerBlock - 1) / threadsPerBlock;
        normalizeCoefficientsKernel << <blocksForNorm, threadsPerBlock >> > (d_a, d_b, d_G, d_D, Ne, Ng); // launching a parallel calculation to normalize coeffs
        LogGpuError("normalizeCoefficientsKernel");

        cudaMemcpy(&d_a[0], &a0, sizeof(double), cudaMemcpyHostToDevice);   // copy the results back to cpu
        
        const int blocksPerGrid = (Ne + threadsPerBlock - 1) / threadsPerBlock;     // modifying the blocks count
        calculateSeriesKernel << <blocksPerGrid, threadsPerBlock >> > (d_Yg, d_x, d_a, d_b, Ne, Ng, w);     // launching a parallel calculation to reconctruct func
        LogGpuError("calculateSeriesKernel");

        cudaEventRecord(stop); // stop the time measurement
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result.executionTimeMs, start, stop);

        // retrieve the results from gpu
        result.calculatedY.resize(Ne);
        result.a_coeffs.resize(Ng + 1);
        result.b_coeffs.resize(Ng + 1);
        cudaMemcpy(result.calculatedY.data(), d_Yg, size_Ne, cudaMemcpyDeviceToHost);
        std::vector<double> G_host(Ng + 1), D_host(Ng + 1);
        cudaMemcpy(G_host.data(), d_G, size_Ng, cudaMemcpyDeviceToHost);
        cudaMemcpy(D_host.data(), d_D, size_Ng, cudaMemcpyDeviceToHost);

        // free the vram
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_G); cudaFree(d_D);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_Yg);

        // final calculations
        result.a_coeffs[0] = a0;
        result.b_coeffs[0] = 0; // b0 always zero
        result.c_amplitudes.resize(Ng + 1);
        result.c_amplitudes[0] = 0;
        for (int k = 1; k <= Ng; ++k) {
            result.a_coeffs[k] = G_host[k] * 2.0 / Ne;
            result.b_coeffs[k] = D_host[k] * 2.0 / Ne;
            result.c_amplitudes[k] = std::sqrt(result.a_coeffs[k] * result.a_coeffs[k] + result.b_coeffs[k] * result.b_coeffs[k]);
        }
        result.isSuccess = true;

    }
    catch (const std::exception& e) {
        result.isSuccess = false;
        result.errorMessage = e.what();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
};