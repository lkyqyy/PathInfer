cmake_minimum_required(VERSION 3.16) 
project(IHC_TensorRT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA) 

# TensorRT/CUDA/openslide
set(TENSORRT_DIR "/home/lk/TensorRT-8.6.1.6")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(OPENSLIDE_DIR "/usr/include/openslide")

# OpenCV
find_package(OpenCV REQUIRED)

include_directories(
    include
    ${TENSORRT_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${OPENSLIDE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${OPENSLIDE_DIR}/lib
)

# 动态库源码
set(SDK_SRC
    src/Infer.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/utils.cpp
    src/preprocess_cuda.cu
    src/inference_pipeline.cpp
    src/global_nms_cuda.cu

)

# 生成动态库
add_library(ihc_trt_sdk SHARED ${SDK_SRC})
target_compile_definitions(ihc_trt_sdk PRIVATE IHC_TRT_SDK_EXPORTS)
target_link_libraries(ihc_trt_sdk
    ${OpenCV_LIBS}
    nvinfer nvinfer_plugin cudart
    openslide          
)

add_executable(IHC_TensorRT_demo main.cpp)
target_link_libraries(IHC_TensorRT_demo ihc_trt_sdk ${OpenCV_LIBS} openslide)
