#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "Infer.h"
#include "utils.h"
#include "inference_pipeline.h"
#include "detect_result.h"



int main() 
{
    auto t_total_start = std::chrono::high_resolution_clock::now();

    std::string engine_path  = "/home/lk/Project/IHC_TensorRT/model/best_dynamic.engine";
    std::string image_path   = "/home/lk/Project/IHC_TensorRT/data/Region_376079759.png";  
    std::string labels_path  = "/home/lk/Project/IHC_TensorRT/classes.txt";
    int patch_size           = 640;
    int overlap              = 30;
    int stride               = patch_size - overlap;
    int batch_size           = 8;


    std::vector<std::string> labels = readClassNames(labels_path);

    cv::Mat full_image = cv::imread(image_path);
    if (full_image.empty()) 
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }
    int64_t w = full_image.cols;
    int64_t h = full_image.rows;
    std::cout << "image size: " << w << "x" << h << std::endl;

   
    IHC_TRT model;
    if (!model.Init(engine_path, 0.25f, 0.45f)) 
    {
        std::cerr << "Failed to initialize TensorRT model." << std::endl;
        return -1;
    }

    
    auto offsets = generatePatchOffsets(w, h, patch_size, stride);
    std::cout << "总Patch数量: " << offsets.size() << "，最后 patch 坐标: "
              << offsets.back().x << ", " << offsets.back().y << std::endl;

    // 多线程推理
    std::vector<DetectResult> global_results;
    std::vector<double> infer_times;
    std::atomic<int> skipped_patch_count{0};

    std::thread t_producer(producer_from_png, std::ref(full_image), std::ref(offsets), patch_size, std::ref(skipped_patch_count));
    std::thread t_consumer(consumer, std::ref(model), std::ref(global_results), std::ref(infer_times));

    t_producer.join();
    t_consumer.join();

    // 推理结果统计
    int total_patch_count = offsets.size();
    int infer_patch_count = total_patch_count - skipped_patch_count;
    int total_detections = global_results.size();

    logPerformanceStats(infer_times, total_patch_count, skipped_patch_count, infer_patch_count, total_detections);

    // 后处理与可视化
    std::vector<DetectResult> nms_results;
    apply_grid_nms(global_results, nms_results, 0.5f, 2048);  

    std::cout << "Grid NMS 后剩余目标数: " << nms_results.size() << std::endl;

    cv::Mat vis_image = full_image.clone();
    drawNMSPoints(nms_results, vis_image, 0.5f);
    cv::imwrite("output/vis_nms_points.jpg", vis_image);

    saveGeoJSON_MultiPoint(nms_results, "output/points_multi.geojson", 0.5f);

    auto t_total_end = std::chrono::high_resolution_clock::now();
    log_time("总推理时间", t_total_start, t_total_end);

    return 0;
}
