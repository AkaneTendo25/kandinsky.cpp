cmake_minimum_required(VERSION 3.14)
project(kandinsky VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(KD_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH "Path to ggml source tree")
get_filename_component(KD_GGML_DIR "${KD_GGML_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT EXISTS "${KD_GGML_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml source tree not found at '${KD_GGML_DIR}'. "
        "Either bundle ggml at '<repo>/ggml' or configure with -DKD_GGML_DIR=<path-to-ggml>.")
endif()

# Backend options (forwarded to ggml)
option(KD_CUDA    "Enable CUDA backend"   OFF)
option(KD_METAL   "Enable Metal backend"  OFF)
option(KD_VULKAN  "Enable Vulkan backend" OFF)
option(KD_OPENCL  "Enable OpenCL backend" OFF)
option(KD_SYCL    "Enable SYCL backend"   OFF)
option(KD_FLASH_ATTN "Enable flash attention" ON)

# Map our options to ggml options
if(KD_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(KD_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(KD_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(KD_OPENCL)
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
endif()
if(KD_SYCL)
    set(GGML_SYCL ON CACHE BOOL "" FORCE)
endif()

# Add ggml as subdirectory
add_subdirectory("${KD_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml-build")

# Collect source files
set(KANDINSKY_SOURCES
    src/core/model_loader.cpp
    src/pipeline.cpp
    src/name_conversion.cpp
)

# Main library
add_library(kandinsky
    ${KANDINSKY_SOURCES}
)

target_include_directories(kandinsky PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    ${KD_GGML_DIR}/include
)

target_link_libraries(kandinsky PUBLIC ggml)

if(MSVC)
    target_compile_definitions(kandinsky PUBLIC NOMINMAX)
endif()

# Backend compile definitions
if(KD_CUDA)
    target_compile_definitions(kandinsky PUBLIC KD_USE_CUDA)
endif()
if(KD_METAL)
    target_compile_definitions(kandinsky PUBLIC KD_USE_METAL)
endif()
if(KD_VULKAN)
    target_compile_definitions(kandinsky PUBLIC KD_USE_VULKAN)
endif()
if(KD_OPENCL)
    target_compile_definitions(kandinsky PUBLIC KD_USE_OPENCL)
endif()
if(KD_SYCL)
    target_compile_definitions(kandinsky PUBLIC KD_USE_SYCL)
endif()
if(KD_FLASH_ATTN)
    target_compile_definitions(kandinsky PUBLIC KD_FLASH_ATTN)
endif()

# CLI executable
add_executable(kandinsky-cli examples/cli/main.cpp)
target_link_libraries(kandinsky-cli PRIVATE kandinsky)
