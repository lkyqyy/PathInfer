#pragma once
# include <fstream>
# include <vector>
#include <opencv2/opencv.hpp>
#include <openslide/openslide.h>
#include "detect_result.h"
#include <opencv2/core.hpp>
#include <utility>

struct pair_hash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};


std::vector<std::string> readClassNames(const std::string& fileName);


std::vector<cv::Point> generatePatchOffsets(int w, int h, int patch_size, int stride);


void log_time(const std::string& desc,
              const std::chrono::high_resolution_clock::time_point& start,
              const std::chrono::high_resolution_clock::time_point& end);

void logPerformanceStats(const std::vector<double>& infer_times,
                         int total_patch_count,     
                         int skipped_patch_count,   
                         int infer_patch_count,     
                         int total_detections) ;


void mergePatchResults(const std::vector<DetectResult>& local_results,
                       const cv::Point& offset,
                       std::vector<DetectResult>& global_results);

 
bool isBlankOrBlackPatch(const cv::Mat& patch,
                         double blank_thresh = 245.0,
                         double black_thresh = 10.0);


void drawNMSPoints(const std::vector<DetectResult>& detections,
                   cv::Mat& image,
                   float conf_thresh = 0.5f,
                   int radius = 10,
                   int thickness = -1);

void saveGeoJSON_MultiPoint(
    const std::vector<DetectResult>& detections,
    const std::string& output_geojson_path,
    float conf_thresh = 0.0f);


// Host 端接口声明
void global_nms_cuda(const std::vector<cv::Rect>& boxes,
                     const std::vector<float>& scores,
                     float iou_thresh,
                     std::vector<int>& keep_indices);



void apply_grid_nms(const std::vector<DetectResult>& input_boxes,
                    std::vector<DetectResult>& output_boxes,
                    float iou_thresh = 0.5f,
                    int grid_size = 1024);
