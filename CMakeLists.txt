cmake_minimum_required(VERSION 3.14)
project(LNOSuperconductivity)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
set(CMAKE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src_tj_double_layer_single_orbital_2d/;${PROJECT_SOURCE_DIR}/src_two_layer_two_orbital_all_dof/")

option(RealCode "tensor element type double or complex" OFF)
if (RealCode)
    add_definitions(-DUSE_REAL=1)
else ()
    add_definitions(-DUSE_COMPLEX=1)
endif ()


# Build type.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

option(QLTEN_TIMING_MODE "Timing mode." OFF)
if (QLTEN_TIMING_MODE)
    add_definitions(-DQLTEN_TIMING_MODE)
endif ()

option(QLMPS_TIMING_MODE "Timing mode." OFF)
if (QLMPS_TIMING_MODE)
    add_definitions(-DQLMPS_TIMING_MODE)
endif ()

option(QLTEN_MPI_TIMING_MODE "Timing mode." OFF)
if (QLTEN_MPI_TIMING_MODE)
    add_definitions(-DQLTEN_MPI_TIMING_MODE)
endif ()

option(QLMPS_MPI_TIMING_MODE "Timing mode." OFF)
if (QLMPS_MPI_TIMING_MODE)
    add_definitions(-DQLMPS_MPI_TIMING_MODE)
endif ()

option(COMPILE_FIX_CODE "compile the fixing code" OFF)

# Compilation and linking control.
set(CMAKE_CXX_STANDARD 17)
add_definitions(-Wall -g)

option(QLMPS_USE_GPU "Use GPU" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/") # HPTT/cutensor

if (QLMPS_USE_GPU)
    add_definitions(-DUSE_GPU=1)

    # Enable CUDA language support
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)  # Modern CUDA Toolkit detection

    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIRS: '${CUDAToolkit_INCLUDE_DIRS}'")

    # Set CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 70 80) # For V100 & A100, adjust if needed
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -Xcompiler=-fPIC")
    # -----------------------------------------------------------
    # Attempt to find cuTENSOR in the same base directory as cuBLAS
    # -----------------------------------------------------------
    # Extract the CUDA Toolkit root path (e.g., /usr/local/cuda or HPC SDK's CUDA dir)
    get_filename_component(CUDA_TOOLKIT_ROOT "${CUDAToolkit_INCLUDE_DIRS}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_TOOLKIT_ROOT}" DIRECTORY)  # Strips off "include"

    # Check if CUDA is part of the NVIDIA HPC SDK (path contains "hpc_sdk")
    if (CUDA_TOOLKIT_ROOT MATCHES "hpc_sdk")
        # Deduce HPC SDK root by removing "/cuda/..." from the path
        string(REGEX REPLACE "/cuda/.*" "" HPC_SDK_ROOT "${CUDA_TOOLKIT_ROOT}")
        set(CUTENSOR_HPC_INC_DIR "${HPC_SDK_ROOT}/math_libs/include")
        set(CUTENSOR_HPC_LIB_DIR "${HPC_SDK_ROOT}/math_libs/lib64")
    else ()
        # For non-HPC SDK installations, check sibling directories
        set(CUTENSOR_HPC_INC_DIR "${CUDA_TOOLKIT_ROOT}/include")
        set(CUTENSOR_HPC_LIB_DIR "${CUDA_TOOLKIT_ROOT}/lib64")
    endif ()

    # Search for cuTENSOR header (account for subdirectory)
    find_path(CUTENSOR_INCLUDE_DIR
            NAMES cutensor.h
            HINTS ${CUTENSOR_HPC_INC_DIR}
            PATH_SUFFIXES cutensor  # Header is in a subdirectory
            DOC "cuTENSOR include directory"
    )

    # Search for cuTENSOR library
    find_library(CUTENSOR_LIBRARY
            NAMES cutensor
            HINTS ${CUTENSOR_HPC_LIB_DIR}
            DOC "cuTENSOR library"
    )

    if (CUTENSOR_INCLUDE_DIR AND CUTENSOR_LIBRARY)
        message(STATUS "Found cuTENSOR in CUDA Toolkit/HPC SDK paths:")
        message(STATUS "  Include dir: ${CUTENSOR_INCLUDE_DIR}")
        message(STATUS "  Library: ${CUTENSOR_LIBRARY}")
    else ()
        message(STATUS "cuTENSOR not found in system. Searching in user directory...")
        # Use custom FindCUTENSOR.cmake from project's cmake directory
        list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")
        find_package(CUTENSOR REQUIRED)
    endif ()


    set(GPU_READY_DIRS
            ${PROJECT_SOURCE_DIR}/src_two_layer_two_orbital_all_dof
            ${PROJECT_SOURCE_DIR}/src_tj_double_layer_single_orbital_2d
            ${PROJECT_SOURCE_DIR}/src_tj_single_layer_single_orbital_2d_anisotropic
            ${PROJECT_SOURCE_DIR}/src_kondo_zigzag_ladder
            ${PROJECT_SOURCE_DIR}/src_kondo_two_layer_2d
            ${PROJECT_SOURCE_DIR}/src_kondo_1d_chain)

    set(GPU_SOURCES "")
    foreach(dir IN LISTS GPU_READY_DIRS)
        file(GLOB_RECURSE dir_sources CONFIGURE_DEPENDS "${dir}/*.cpp")
        list(APPEND GPU_SOURCES ${dir_sources})
    endforeach()
    list(REMOVE_DUPLICATES GPU_SOURCES)
    if (GPU_SOURCES)
        message(STATUS "GPU_SOURCES=" ${GPU_SOURCES})
        set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
    endif ()
else ()
    find_package(hptt)
    message(STATUS "HPTT Lib : " "${hptt_LIBRARY}")
endif ()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(BLA_VENDOR Intel10_64lp)   #FOR VENDOR CHOICE: https://cmake.org/cmake/help/latest/module/FindBLAS.html
    set(BLAS_INCLUDE_DIR "$ENV{MKLROOT}/include")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64") # (e.g., Apple Silicon)
    add_definitions(-DUSE_OPENBLAS)
    set(BLA_VENDOR OpenBLAS)
    set(OpenBLAS_ROOT "/opt/homebrew/opt/openblas/") #work for homebrew installed openblas in MacOSX
    set(Lapack_ROOT "/opt/homebrew/opt/lapack")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OpenBLAS_ROOT} ${Lapack_ROOT}) # For find
    set(BLAS_INCLUDE_DIR "${OpenBLAS_ROOT}/include")
endif ()

FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

find_path(TENSOR_HEADER_PATH "qlten")
find_path(MPS_HEADER_PATH "qlmps")


# Intel MKL multi-thread may conflict with other compiler's OpenMP
# Close OpenMP if using MKL with other compiler
if (NOT BLA_VENDOR STREQUAL "Intel10_64lp" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    find_package(OpenMP REQUIRED)
    link_libraries(OpenMP::OpenMP_CXX)
endif ()

include_directories(
        ${BLAS_INCLUDE_DIR}
        ${MPI_CXX_HEADER_DIR}
        ${TENSOR_HEADER_PATH}
        ${MPS_HEADER_PATH})

link_libraries(BLAS::BLAS
        LAPACK::LAPACK
        MPI::MPI_CXX)

if (NOT QLMPS_USE_GPU)
    include_directories(${hptt_INCLUDE_DIR})
    link_libraries(${hptt_LIBRARY})
else ()
    include_directories(${CUTENSOR_INCLUDE_DIR})
    link_libraries(
            CUDA::cublas_static
            CUDA::cusolver_static
            ${CUTENSOR_LIBRARY}
    )
endif ()

#
#add_executable(mpogen src_tj_double_layer_single_orbital_2d/mpogen.cpp src_tj_double_layer_single_orbital_2d/operators.cpp)
#add_executable(vmps src_tj_double_layer_single_orbital_2d/vmps.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp src_tj_double_layer_single_orbital_2d/operators.cpp)
add_executable(dmrg_single_orbital src_tj_double_layer_single_orbital_2d/dmrg_single_orbital_double_layer.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
#add_executable(measure1 src_tj_double_layer_single_orbital_2d/measure1.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp src_tj_double_layer_single_orbital_2d/operators.cpp)
#add_executable(measure2 src_tj_double_layer_single_orbital_2d/measure2.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp src_tj_double_layer_single_orbital_2d/operators.cpp)
#add_executable(measureSC src_tj_double_layer_single_orbital_2d/measureSC.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp src_tj_double_layer_single_orbital_2d/operators.cpp)

add_library(4band_common_objects OBJECT src_two_layer_two_orbital_all_dof/myutil.cpp)

add_executable(dmrg4band src_two_layer_two_orbital_all_dof/dmrg.cpp $<TARGET_OBJECTS:4band_common_objects>)
add_executable(measure4band1 src_two_layer_two_orbital_all_dof/measure1.cpp $<TARGET_OBJECTS:4band_common_objects>)
add_executable(measure4bandSC src_two_layer_two_orbital_all_dof/measureSC_fix_ref.cpp $<TARGET_OBJECTS:4band_common_objects>)

add_executable(dmrg_ani_tJ src_tj_single_layer_single_orbital_2d_anisotropic/dmrg_single_orbital_single_layer.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(measure1_ani_tJ src_tj_single_layer_single_orbital_2d_anisotropic/measure1.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)

add_executable(kondo_chain_vmps src_kondo_1d_chain/vmps.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_ladder_vmps src_kondo_zigzag_ladder/vmps.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_ladder_measure src_kondo_zigzag_ladder/measure.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_ladder_vmpsJk0 src_kondo_zigzag_ladder/vmps_Jk0.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_ladder_dmrgJk0 src_kondo_zigzag_ladder/dmrg_Jk0.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_ladder_conventional_square_vmps src_kondo_zigzag_ladder/vmps_conventional_square.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(visualize_tilted_lattice src_kondo_zigzag_ladder/visualize_tilted_lattice.cpp)

add_executable(kondo_two_layer_vmps src_kondo_two_layer_2d/vmps_conventional_square.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
# Kondo two-layer fix programs
add_executable(kondo_two_layer_mpo_gen src_kondo_two_layer_2d/mpo_gen.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_two_layer_fix_env src_kondo_two_layer_2d/fix_env_kondo.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_two_layer_fix_mps src_kondo_two_layer_2d/fix_mps_kondo.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_two_layer_fix_update_lenv src_kondo_two_layer_2d/fix_update_lenv_kondo.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_two_layer_move_center src_kondo_two_layer_2d/move_center.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_two_layer_fix_mps2 src_kondo_two_layer_2d/fix_mps2_kondo.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)
add_executable(kondo_two_layer_measure_sp src_kondo_two_layer_2d/measure_sp.cpp src_tj_double_layer_single_orbital_2d/myutil.cpp)