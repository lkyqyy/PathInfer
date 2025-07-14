#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "iou_utils.cuh"
#include "postprocess.h"  


__global__ void global_nms_kernel(const cv::Rect* boxes, const float* scores,
                                  int num_boxes, float iou_thresh, int* keep_flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes) return;

    const cv::Rect cur_box = boxes[i];
    float cur_score = scores[i];

    for (int j = 0; j < num_boxes; ++j) {
        if (j == i) continue;
        if (scores[j] >= cur_score) {
            if (IoU(cur_box, boxes[j]) > iou_thresh) {
                keep_flags[i] = 0;
                return;
            }
        }
    }
    keep_flags[i] = 1;
}


void launch_global_nms_kernel(const cv::Rect* d_boxes, const float* d_scores, int* d_keep_flags,
                              int num_boxes, float iou_thresh, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_boxes + threads - 1) / threads;
    global_nms_kernel<<<blocks, threads, 0, stream>>>(d_boxes, d_scores, num_boxes, iou_thresh, d_keep_flags);
}


void nms_cuda(const std::vector<cv::Rect>& boxes,
              const std::vector<float>& scores,
              float iou_thresh,
              std::vector<int>& keep_indices)
{
    int num_boxes = boxes.size();
    if (num_boxes == 0) return;

    cv::Rect* d_boxes;
    float* d_scores;
    int* d_keep_flags;

    cudaMalloc(&d_boxes, sizeof(cv::Rect) * num_boxes);
    cudaMalloc(&d_scores, sizeof(float) * num_boxes);
    cudaMalloc(&d_keep_flags, sizeof(int) * num_boxes);

    cudaMemcpy(d_boxes, boxes.data(), sizeof(cv::Rect) * num_boxes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores.data(), sizeof(float) * num_boxes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    launch_global_nms_kernel(d_boxes, d_scores, d_keep_flags, num_boxes, iou_thresh, stream);
    cudaStreamSynchronize(stream);

    std::vector<int> flags(num_boxes);
    cudaMemcpy(flags.data(), d_keep_flags, sizeof(int) * num_boxes, cudaMemcpyDeviceToHost);


    keep_indices.clear();
    for (int i = 0; i < num_boxes; ++i) {
        if (flags[i]) keep_indices.push_back(i);
    }

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_keep_flags);
    cudaStreamDestroy(stream);
}
