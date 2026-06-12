cmake_minimum_required(VERSION 3.10)
project(libai_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Compile with CUDA support" OFF)
option(USE_AVX2 "Compile with AVX2 support" ON)

# If llama.cpp is not installed globally, users should provide LLAMA_PATH
set(LLAMA_PATH "" CACHE PATH "Path to llama.cpp source/build directory")

find_library(LLAMA_LIB llama HINTS ${LLAMA_PATH})
find_path(LLAMA_INCLUDE llama.h HINTS ${LLAMA_PATH} PATH_SUFFIXES include)

if(NOT LLAMA_LIB OR NOT LLAMA_INCLUDE)
    message(WARNING "llama.cpp libraries not found. Please set LLAMA_PATH.")
endif()

include_directories(${LLAMA_INCLUDE})

add_library(ai_core SHARED inference.cpp inference.h)
target_link_libraries(ai_core ${LLAMA_LIB})

if(USE_AVX2)
    if(MSVC)
        target_compile_options(ai_core PRIVATE /arch:AVX2)
    else()
        target_compile_options(ai_core PRIVATE -mavx2)
    endif()
endif()

if(USE_CUDA)
    add_definitions(-DGGML_USE_CUDA)
    # CUDA linking would go here if specific CUDA libs are needed beyond what llama.cpp provides
endif()
