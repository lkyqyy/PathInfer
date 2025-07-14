#pragma once
#include <opencv2/opencv.hpp>
#include <vector>


struct DetectResult {
    int classId;
    float conf;
    cv::Rect box;
};

struct PatchData 
{
    cv::Mat image;
    cv::Point offset;
};