add_executable(simple_cutlass_gemm simple_cutlass_gemm.cu)
set_target_properties(simple_cutlass_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

target_compile_options(simple_cutlass_gemm PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>: --ftz=true -O3 -DNDEBUG --expt-relaxed-constexpr>
)

target_link_libraries(simple_cutlass_gemm PRIVATE CutlassGemm)
