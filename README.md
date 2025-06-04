# CUTLASS Experiments

Cutlass provides templates/abstractions for high-performance GEMM and CONV.

I try out a number of different techniques, and document as much as I can on the way.

There are different levels of API, related to the GPU memory hierarchy of NVIDIA GPUs.

- Threadblocklevel abstractions, for matrix multiply-accumulate operations.
- Warp level, for matrix multiply-accumulate operations.
- Epilogue components, for tensor ops/saving
- Loading/saving utilities.

### CUTE

- A template library for tensors, built on top of cutlass, and provides more flexibility for specifying the layout of tensors with reference to GPU memory hierarchies.
- Introduced in latest major release of CUTLASS (3.0)


### Run Example Scipts

Dependencies:
- CUDA
- clang format (optional)

```bash
# configure
cmake --preset nvidia-release


# Examples:

# 1. Simple CUTLASS Based GEMM based on the tutorial

# build
cmake --build --preset nvidia-release --target simple_cutlass_gemm
# run the example directly and build
cmake --build --preset nvidia-release --target run_simple_cutlass_gemm

# Other Commands:
# 1. Format
cmake --build --preset nvidia-release --target format
```