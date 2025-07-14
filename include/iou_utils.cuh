#pragma once
#include <opencv2/opencv.hpp>

// 在多个 .cu 中使用的通用 device 函数
inline __device__ float IoU(const cv::Rect& a, const cv::Rect& b) 
{
    int x1 = max(a.x, b.x);
    int y1 = max(a.y + a.width, b.x + b.width);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);
    int inter = max(0, x2 - x1) * max(0, y2 - y1);
    int area_a = a.width * a.height;
    int area_b = b.width * b.height;
    return inter / float(area_a + area_b - inter + 1e-6f);  // 防止除以0
}
