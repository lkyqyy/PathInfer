#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include "preprocess.h"

__global__ void preprocess_kernel(const unsigned char* input, float* output,
                                  int input_w, int input_h, int output_w, int output_h,
                                  int pad_w, int pad_h, float scale,
                                  int batch_idx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_w || y >= output_h) return;

    int out_idx = batch_idx * 3 * output_h * output_w;

    int in_x = (x - pad_w) / scale;
    int in_y = (y - pad_h) / scale;

    for (int ch = 0; ch < 3; ++ch) {
        float val = 0.f;
        if (in_x >= 0 && in_x < input_w && in_y >= 0 && in_y < input_h) {
            int in_idx = (in_y * input_w + in_x) * 3;
            unsigned char pix = input[in_idx + (2 - ch)]; // BGR → RGB
            val = pix / 255.0f;
        }
        int out_pos = out_idx + ch * output_h * output_w + y * output_w + x;
        output[out_pos] = val;
    }
}


void preprocessBatchCUDA(const std::vector<cv::Mat>& images,
                         float* dst_device,
                         int batch_size,
                         int input_w,
                         int input_h,
                         cudaStream_t stream,
                         std::vector<float>& scales,
                         std::vector<int>& pads_w,
                         std::vector<int>& pads_h)
{
    size_t img_size = input_w * input_h * 3;  
    unsigned char* d_image = nullptr;
    cudaMalloc(&d_image, img_size);

    scales.clear();
    pads_w.clear();
    pads_h.clear();

    for (int i = 0; i < batch_size; ++i)
    {
        float scale;
        int pad_w, pad_h;

        cv::Mat resized = letterboxResize(images[i], input_w, input_h, scale, pad_w, pad_h);

        cudaMemcpyAsync(d_image, resized.data, img_size, cudaMemcpyHostToDevice, stream);

        dim3 block(32, 32);
        dim3 grid((input_w + 31) / 32, (input_h + 31) / 32);
        preprocess_kernel<<<grid, block, 0, stream>>>
        (
            d_image, dst_device, resized.cols, resized.rows,
            input_w, input_h, pad_w, pad_h, scale, i
        );

        scales.push_back(scale);
        pads_w.push_back(pad_w);
        pads_h.push_back(pad_h);
    }

    cudaFree(d_image);
}
