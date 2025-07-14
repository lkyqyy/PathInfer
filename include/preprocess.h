#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void preprocessBatchCUDA(const std::vector<cv::Mat>& images,
                         float* dst_device,
                         int batch_size,
                         int input_w,
                         int input_h,
                         cudaStream_t stream,
                         std::vector<float>& scales,
                         std::vector<int>& pads_w,
                         std::vector<int>& pads_h);


cv::Mat letterboxResize(const cv::Mat& image, int target_w, int target_h, float& scale, int& pad_w, int& pad_h);


cv::Mat convertToBlob(const cv::Mat& image, int input_w, int input_h);