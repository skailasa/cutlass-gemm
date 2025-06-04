#pragma once
#include <cuda_runtime.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

CUTLASS_HOST_DEVICE
float test_dot(float a, float b) {
  return a * b;
}

__global__ void device_add(float* x, float* y, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + y[i];
}

__global__ void device_add_half(const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];  // uses cutlass::half_t operator+
    }
}

