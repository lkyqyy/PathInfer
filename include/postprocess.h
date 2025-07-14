#pragma once
#include <opencv2/opencv.hpp>
#include "Infer.h"


void decodeOutputs(const float* prob, int output_h, int output_w,
                   float conf_thresh, float nms_thresh,
                   float scale, int pad_w, int pad_h,
                   int input_w, int input_h,
                   std::vector<DetectResult>& results);

void nms_cuda(const std::vector<cv::Rect>& boxes,
              const std::vector<float>& scores,
              float iou_thresh,
              std::vector<int>& keep_indices);