#pragma once

#include <vector>
#include <string>

using namespace std;

#ifdef FOURIERLIB_EXPORTS
#define CUDA_API __declspec(dllexport)
#else
#define CUDA_API __declspec(dllimport)
#endif

struct Params { // parameters received from ui
    int numHarmonics; // number of fourier harmonics
};

struct Result {     // parameters returned to ui
    vector<double> calculatedY;    // calculated series points (Yg)
    vector<double> c_amplitudes;    // garm amplitudes (c)
    vector<double> a_coeffs;    // coeffs (a)
    vector<double> b_coeffs;    // coeffs (b)
    float executionTimeMs;  // telemetry
    bool isSuccess; // another telemetry
    string errorMessage;   // another another telemetry (self-explanatory)
};

class CUDA_API FourierCudaCalculator {
public:
    FourierCudaCalculator();    // constructor
    ~FourierCudaCalculator();   // destructor

    std::vector<std::string> GetAvailableDevices(); // get the array of cuda-compatible gpus available now
    bool SelectDevice(int deviceId); // select the device by it's id in the array above

    Result Calculate(
        const Params& params,
        const std::vector<double>& x_values, // tabulated x and y from function (received from ui)
        const std::vector<double>& y_values
    );

private:
    int selectedDeviceIndex; // selected device id
};

