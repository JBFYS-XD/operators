add_executable(test_frame_sgemm_cuda-core "tester/test_frame_sgemm_cuda-core.cu")
target_link_libraries(test_frame_sgemm_cuda-core MPI::MPI_CXX CUDA::cudart CUDA::cublas)
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(test_frame_sgemm_cuda-core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()


