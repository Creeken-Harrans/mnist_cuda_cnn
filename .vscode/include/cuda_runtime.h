// Stub for VSCode IntelliSense — not used by nvcc.
#pragma once

#ifdef __INTELLISENSE__
// ---- CUDA function qualifiers (silence __global__ / __device__ / __host__) ----
#define __global__
#define __device__
#define __host__
#define __restrict__ __restrict__
#define __forceinline__ inline

// ---- atomicAdd for device code (stub) ----
inline float atomicAdd(float* addr, float val) { float old = *addr; *addr += val; return old; }
inline int   atomicAdd(int*   addr, int   val) { int   old = *addr; *addr += val; return old; }

// ---- CUDA error codes ----
typedef int cudaError_t;
enum {
    cudaSuccess = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
};

// ---- CUDA structs ----
struct cudaDeviceProp { char name[256]; };

// ---- CUDA runtime API ----
inline const char* cudaGetErrorString(cudaError_t) { return ""; }
inline cudaError_t cudaMalloc(void** p, size_t n)      { *p = nullptr; (void)n; return cudaSuccess; }
inline cudaError_t cudaFree(void* p)                    { (void)p; return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, int) { (void)d; (void)s; (void)n; return cudaSuccess; }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { (void)p; (void)v; (void)n; return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize()              { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int)                   { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { (void)p; return cudaSuccess; }

#endif // __INTELLISENSE__
