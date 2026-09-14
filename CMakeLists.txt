cmake_minimum_required(VERSION 3.22)

project(GPU_M2D VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

option(GPU_M2D_BUILD_TESTS "Build GPU_M2D validation tests" ON)

add_library(gpu_m2d_semantic STATIC
    src/semantic/tensor_mapping.cpp
)
target_include_directories(gpu_m2d_semantic PUBLIC include)
target_compile_options(gpu_m2d_semantic PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_library(gpu_m2d_injector STATIC
    src/injector/device_bit_injector.cu
)
target_include_directories(gpu_m2d_injector PUBLIC include)
target_link_libraries(gpu_m2d_injector PUBLIC CUDA::cudart)
set_target_properties(gpu_m2d_injector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if(GPU_M2D_BUILD_TESTS)
    enable_testing()

    add_executable(test_semantic_mapper tests/test_semantic_mapper.cpp)
    target_link_libraries(test_semantic_mapper PRIVATE gpu_m2d_semantic)
    add_test(NAME semantic_mapper COMMAND test_semantic_mapper)

    add_executable(test_cuda_injector tests/test_cuda_injector.cu)
    target_link_libraries(test_cuda_injector PRIVATE
        gpu_m2d_semantic
        gpu_m2d_injector
        CUDA::cudart
    )
    add_test(NAME cuda_injector_device0 COMMAND test_cuda_injector --device 0)
endif()
