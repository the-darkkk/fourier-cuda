#define USE_SAFE_ATOMICS 1 // directive that switches to looped implementation of atomicAdd for doubles.
// recommended to use for nvidia maxwell architecture and any compatibility layers for launching on non-nvidia hardware

#if defined(USE_SAFE_ATOMICS) // basically uses uns long long int insted of doubles
__device__ double safeAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#else
__device__ double safeAtomicAdd(double* address, double val)
{
    return atomicAdd(address, val);
}
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

// step 1 of calculating fourier sequence - calculating the coeffs
__global__ void calculateCoefficientsKernel(double* d_G, double* d_D, const double* d_x, const double* d_y, int Ne, int Ng, double w) {
    int k = blockIdx.x + 1; // number of garmonic
    if (k > Ng) return;

    // local data inside of each thread
    int tid = threadIdx.x;
    double G_private = 0.0;
    double D_private = 0.0;

    // calculation
    for (int i = tid; i < Ne; i += blockDim.x) {
        const double S = k * w * d_x[i];
        G_private += d_y[i] * cos(S);
        D_private += d_y[i] * sin(S);
    }

    // adding the calculated data together from all threads
    safeAtomicAdd(&d_G[k], G_private);
    safeAtomicAdd(&d_D[k], D_private);
}

// step 2 - reconstructing the function from given garmonics
__global__ void calculateSeriesKernel(double* d_Yg, const double* d_x, const double* d_a, const double* d_b, int Ne, int Ng, double w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= Ne) return;

    const double a0 = d_a[0];
    const double D_val = d_x[i] * w;
    double S = 0.0;

    for (int k = 1; k <= Ng; ++k) {
        const double KOM = k * D_val;
        S += d_b[k] * sin(KOM) + d_a[k] * cos(KOM);
    }
    d_Yg[i] = a0 + S;
};

// normalizing coefficients between step 1 and step 2
__global__ void normalizeCoefficientsKernel(double* d_a, double* d_b, const double* d_G, const double* d_D, int Ne, int Ng) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1; 
    if (k > Ng) return;
    d_a[k] = d_G[k] * 2.0 / Ne;
    d_b[k] = d_D[k] * 2.0 / Ne;
}