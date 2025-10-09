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