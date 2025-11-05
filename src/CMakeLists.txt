add_executable(test_frame "test_frame.cu")
target_link_libraries(test_frame MPI::MPI_CXX CUDA::cudart CUDA::cublas)
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(test_frame PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()


