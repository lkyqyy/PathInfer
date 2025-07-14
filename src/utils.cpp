# include <fstream>
# include <vector>
#include <iostream>
#include <numeric>  
#include <cstdlib>
#include "utils.h"
#include "detect_result.h"
#include <nlohmann/json.hpp>


#include <cuda_runtime.h>
#include <iomanip>
#include <filesystem>
#include <random> 
#include <unordered_map> 


using json = nlohmann::json;

static std::vector<cv::Scalar> class_colors = {
    cv::Scalar(0, 0, 255),     
    cv::Scalar(0, 255, 0),    
    cv::Scalar(255, 0, 0),     
};


std::vector<std::string> readClassNames(const std::string& filename) 
{
    std::vector<std::string> classNames;
    std::ifstream fp(filename);
    if (!fp.is_open()) 
    {
        std::cerr << "Could not open class names file: " << filename << std::endl;
        exit(-1);
    }
    std::string name;
    while (std::getline(fp, name)) 
    {
        if (!name.empty()) classNames.push_back(name);
    }
    return classNames;
}


std::vector<cv::Point> generatePatchOffsets(int w, int h, int patch_size, int stride) {
    std::vector<cv::Point> offsets;
    for (int y = 0; y + patch_size <= h; y += stride) {
        for (int x = 0; x + patch_size <= w; x += stride) {
            offsets.emplace_back(x, y);
        }
    }
    return offsets;
}



void log_time(const std::string& desc,
              const std::chrono::high_resolution_clock::time_point& start,
              const std::chrono::high_resolution_clock::time_point& end) {
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << " " << desc << ": " << ms << " ms" << std::endl;
}

void logPerformanceStats(const std::vector<double>& infer_times,
                         int total_patch_count,    
                         int skipped_patch_count,   
                         int infer_patch_count,     
                         int total_detections)      
{
    if (infer_times.empty()) return;

    double avg = std::accumulate(infer_times.begin(), infer_times.end(), 0.0) / infer_times.size();
    double max = *std::max_element(infer_times.begin(), infer_times.end());
    double min = *std::min_element(infer_times.begin(), infer_times.end());

    std::cout << "推理耗时统计: avg=" << avg
              << " ms, min=" << min
              << " ms, max=" << max << " ms" << std::endl;

    std::cout << "原始 patch 总数: " << total_patch_count << std::endl;
    std::cout << "跳过空白或全黑 patch 数: " << skipped_patch_count << std::endl;
    std::cout << "实际推理 patch 数: " << infer_patch_count << std::endl;
    std::cout << "总检测目标数: " << total_detections << std::endl;
}

void saveResultsToGeoJSON(const std::string& filename,
                          const std::vector<DetectResult>& results,
                          const std::vector<std::string>& class_names) 
    {
        nlohmann::json geojson;
        geojson["type"] = "FeatureCollection";
        for (const auto& r : results) 
        {
            nlohmann::json feature;
            feature["type"] = "Feature";
            feature["properties"]["category_id"] = r.classId;
            feature["properties"]["category_name"] = class_names[r.classId];
            feature["properties"]["confidence"] = r.conf;

            int x = r.box.x;
            int y = r.box.y;
            int w = r.box.width;
            int h = r.box.height;

            feature["geometry"]["type"] = "Polygon";
            feature["geometry"]["coordinates"] = 
            {
                {
                {x, y},
                {x + w, y},
                {x + w, y + h},
                {x, y + h},
                {x, y}
                }
            };
        geojson["features"].push_back(feature);
        }

    std::ofstream out(filename);
    out << geojson.dump(2);
}

void mergePatchResults(const std::vector<DetectResult>& local_results,
                       const cv::Point& offset,
                       std::vector<DetectResult>& global_results) 
                       {
                        for (const auto& r : local_results) 
                        {
                            DetectResult dr = r;
                            if (dr.box.x < 0 || dr.box.x >= 640 || dr.box.y < 0 || dr.box.y >= 640) 
                            {
                             std::cerr << "模型输出框越界：局部坐标 (" << dr.box.x << ", " << dr.box.y << ")\n";
                            }
                            dr.box.x += offset.x;
                            dr.box.y += offset.y;
                            if (dr.box.x < offset.x || dr.box.x > offset.x + 640 ||
                                dr.box.y < offset.y || dr.box.y > offset.y + 640) 
                                {
                                    std::cerr << " dr.box=(" << dr.box.x << ", " << dr.box.y
                                    << ") offset=(" << offset.x << ", " << offset.y << ")\n";
                                }
                                global_results.push_back(dr);
                        }
                       }

bool isBlankOrBlackPatch(const cv::Mat& patch, double blank_thresh, double black_thresh) 
{
    if (patch.empty()) return true;

    cv::Mat gray;
    cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);


    cv::Scalar mean_val = cv::mean(gray);
    double mean_intensity = mean_val[0];

    return (mean_intensity > blank_thresh || mean_intensity < black_thresh);
}



void drawNMSPoints(const std::vector<DetectResult>& detections,
                   cv::Mat& image,
                   float conf_thresh,
                   int radius,
                   int thickness)
{
    for (const auto& det : detections) {
        if (det.conf < conf_thresh) continue;

        float cx = det.box.x + det.box.width / 2.0f;
        float cy = det.box.y + det.box.height / 2.0f;
        cv::Point center(static_cast<int>(cx), static_cast<int>(cy));

        cv::Scalar color = class_colors[det.classId % class_colors.size()];
        cv::circle(image, center, radius, color, thickness);
    }
}


std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);

    const char* hex_chars = "0123456789abcdef";
    std::string uuid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";

    for (auto& c : uuid) {
        if (c == 'x') c = hex_chars[dis(gen)];
        else if (c == 'y') c = hex_chars[(dis(gen) & 0x3) | 0x8];
    }
    return uuid;
}

void saveGeoJSON_MultiPoint(
    const std::vector<DetectResult>& detections,
    const std::string& output_geojson_path,
    float conf_thresh)
{
    json geojson;
    geojson["type"] = "FeatureCollection";
    geojson["features"] = json::array();

    json feature;
    feature["type"] = "Feature";
    feature["id"] = generate_uuid();

    json coords = json::array();
    for (const auto& det : detections) {
        if (det.conf < conf_thresh) continue;

        float cx = det.box.x + det.box.width / 2.0f;
        float cy = det.box.y + det.box.height / 2.0f;

        coords.push_back({cx, cy});
    }

    feature["geometry"] = {
        {"type", "MultiPoint"},
        {"coordinates", coords}
    };

    feature["properties"] = {
        {"objectType", "annotation"}
    };

    geojson["features"].push_back(feature);

    std::filesystem::create_directories(std::filesystem::path(output_geojson_path).parent_path());

    std::ofstream ofs(output_geojson_path);
    ofs << std::setw(2) << geojson << std::endl;
    std::cout << "MultiPoint GeoJSON written to: " << output_geojson_path << std::endl;
}


void launch_global_nms_kernel(const cv::Rect* d_boxes, const float* d_scores, int* d_keep_flags,
                              int num_boxes, float iou_thresh, cudaStream_t stream = 0);

void global_nms_cuda(const std::vector<cv::Rect>& boxes,
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

    launch_global_nms_kernel(d_boxes, d_scores, d_keep_flags, N, iou_thresh);

    std::vector<int> keep_flags(N);
    cudaMemcpy(keep_flags.data(), d_keep_flags, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        if (keep_flags[i]) keep_indices.push_back(i);

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_keep_flags);
}



static std::pair<int, int> getGridKey(const cv::Rect& box, int grid_size) {
    return {box.x / grid_size, box.y / grid_size};
}

void apply_grid_nms(const std::vector<DetectResult>& input_boxes,
                    std::vector<DetectResult>& output_boxes,
                    float iou_thresh,
                    int grid_size)
{
    using GridKey = std::pair<int, int>;
    std::unordered_map<GridKey, std::vector<int>, pair_hash> grid_map;


    for (int i = 0; i < input_boxes.size(); ++i) {
        GridKey key = getGridKey(input_boxes[i].box, grid_size);
        grid_map[key].push_back(i);
    }

    for (const auto& kv : grid_map) {
        const std::vector<int>& indices = kv.second;

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<DetectResult> group;

        for (int idx : indices) {
            boxes.push_back(input_boxes[idx].box);
            scores.push_back(input_boxes[idx].conf);
            group.push_back(input_boxes[idx]);
        }

        std::vector<int> keep;
        global_nms_cuda(boxes, scores, iou_thresh, keep);

        for (int ki : keep) {
            output_boxes.push_back(group[ki]);
        }
    }
}
