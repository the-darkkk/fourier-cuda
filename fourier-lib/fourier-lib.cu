#define _USE_MATH_DEFINES
#include <cmath>
#include "fourier-lib.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void calculateCoefficientsKernel(double*, double*, const double*, const double*, int, int, double);
__global__ void calculateSeriesKernel(double*, const double*, const double*, const double*, int, int, double);

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

Result FourierCudaCalculator::Calculate(
    const Params& params,
    const std::vector<double>& x_values,
    const std::vector<double>& y_values)
{
    Result result;

    const int Ne = x_values.size();
    const int Ng = params.numHarmonics;
    const double Tp = x_values.back() - x_values.front();
    const double w = 2.0 * M_PI / Tp;

    double a0_sum = 0.0;
    for (int i = 0; i < Ne; ++i) a0_sum += y_values[i];
    double a0 = a0_sum / Ne;

    double* d_x, * d_y, * d_a, * d_b, * d_Yg;
    size_t size_Ne = Ne * sizeof(double);
    size_t size_Ng = (Ng + 1) * sizeof(double);
    cudaMalloc(&d_x, size_Ne);
    cudaMalloc(&d_y, size_Ne);
    cudaMalloc(&d_a, size_Ng);
    cudaMalloc(&d_b, size_Ng);
    cudaMalloc(&d_Yg, size_Ne);

    cudaMemcpy(d_x, x_values.data(), size_Ne, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_values.data(), size_Ne, cudaMemcpyHostToDevice);
    cudaMemset(d_a, 0, size_Ng);
    cudaMemset(d_b, 0, size_Ng);

    int threadsPerBlockCoeffs = 256;
    calculateCoefficientsKernel << <Ng, threadsPerBlockCoeffs >> > (d_a, d_b, d_x, d_y, Ne, Ng, w);

    cudaMemcpy(&d_a[0], &a0, sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlockReconstruct = 256;
    int blocksPerGridReconstruct = (Ne + threadsPerBlockReconstruct - 1) / threadsPerBlockReconstruct;
    calculateSeriesKernel << <blocksPerGridReconstruct, threadsPerBlockReconstruct >> > (d_Yg, d_x, d_a, d_b, Ne, Ng, w);

    result.calculatedY.resize(Ne);
    result.a_coeffs.resize(Ng + 1);
    result.b_coeffs.resize(Ng + 1);
    cudaMemcpy(result.calculatedY.data(), d_Yg, size_Ne, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.a_coeffs.data(), d_a, size_Ng, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.b_coeffs.data(), d_b, size_Ng, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_Yg);

    result.c_amplitudes.resize(Ng + 1);
    for (int k = 1; k <= Ng; ++k) {
        result.c_amplitudes[k] = std::sqrt(result.a_coeffs[k] * result.a_coeffs[k] + result.b_coeffs[k] * result.b_coeffs[k]);
    }

    result.isSuccess = true;
    return result;
}