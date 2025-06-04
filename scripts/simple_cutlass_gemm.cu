#include <iostream>
#include <vector>
#include "CutlassGemm.cuh"
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <iterator>
#include <algorithm>

int main() {
    float x = 3.0f, y = 4.0f;
    float z = test_dot(x, y);
    std::cout << "x * y = " << z << std::endl;


    dim3 gridDim = (1);
    dim3 blockDim = (32);

    int N = 32;

    using half = cutlass::half_t;

    std::vector<half> A(N, half(1.0f));
    std::vector<half> B(N, half(2.0f));
    std::vector<half> C(N, half(0.0f));

    half *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, N * sizeof(half));
    cudaMalloc(&B_d, N * sizeof(half));
    cudaMalloc(&C_d, N * sizeof(half));

    cudaMemcpy(A_d, A.data(), N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    device_add_half<<<gridDim, blockDim>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), C_d, N * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << static_cast<float>(C[i]) << " ";
    }
    std::cout << std::endl;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}