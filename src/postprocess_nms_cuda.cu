#include "postprocess.h"
#include <cuda_runtime.h>
#include "iou_utils.cuh"


__global__ void nms_kernel(const cv::Rect* boxes, const float* scores, int num_boxes, float iou_thresh, int* keep_flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes) return;

    for (int j = 0; j < num_boxes; ++j) {
        if (j == i) continue;
        if (scores[j] > scores[i]) {
            if (IoU(boxes[i], boxes[j]) > iou_thresh) {
                keep_flags[i] = 0;
                return;
            }
        }
    }
    keep_flags[i] = 1;
}

void nms_cuda(const std::vector<cv::Rect>& boxes,
              const std::vector<float>& scores,
              float iou_thresh,
              std::vector<int>& keep_indices)
{
    int N = boxes.size();
    if (N == 0) return;

    cv::Rect* d_boxes;
    float* d_scores;
    int* d_keep_flags;

    cudaMalloc(&d_boxes, N * sizeof(cv::Rect));
    cudaMalloc(&d_scores, N * sizeof(float));
    cudaMalloc(&d_keep_flags, N * sizeof(int));

    cudaMemcpy(d_boxes, boxes.data(), N * sizeof(cv::Rect), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    nms_kernel<<<blocks, threads>>>(d_boxes, d_scores, N, iou_thresh, d_keep_flags);

    std::vector<int> keep_flags(N);
    cudaMemcpy(keep_flags.data(), d_keep_flags, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        if (keep_flags[i]) keep_indices.push_back(i);
    }

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_keep_flags);
}
