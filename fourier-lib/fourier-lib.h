#pragma once

#include <vector>
#include <string>

#ifdef FOURIERLIB_EXPORTS
#define CUDA_API __declspec(dllexport)
#else
#define CUDA_API __declspec(dllimport)
#endif