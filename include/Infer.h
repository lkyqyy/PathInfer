#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "detect_result.h"


// 导出宏定义
#ifdef _WIN32
  #ifdef IHC_TRT_SDK_EXPORTS
    #define IHC_TRT_API __declspec(dllexport)
  #else
    #define IHC_TRT_API __declspec(dllimport)
  #endif
#else
  #define IHC_TRT_API
#endif

class IHC_TRT_API IHC_TRT
{
public:
    IHC_TRT();
    ~IHC_TRT();

    bool Init(const std::string& engine_path, float conf_thresh = 0.25f, float score_thresh = 0.45f);

    bool Inference(const cv::Mat& image, std::vector<DetectResult>& results);

    bool InferenceBatch(const std::vector<cv::Mat>& images, std::vector<std::vector<DetectResult>>& batch_results);
    
private:
    struct Impl;
    Impl* pimpl_;
};

