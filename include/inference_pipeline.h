#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <openslide/openslide.h>
#include "Infer.h"
#include "detect_result.h"
#include <condition_variable>
#include <queue>
#include <mutex>



extern std::queue<PatchData> patch_queue;
extern std::mutex queue_mutex;
extern std::condition_variable queue_cv;
extern bool producer_done;

void consumer(IHC_TRT& model, std::vector<DetectResult>& global_results, std::vector<double>& infer_times);

void producer_from_png(const cv::Mat& full_image, const std::vector<cv::Point>& offsets,
                       int patch_size, std::atomic<int>& skipped_patch_count);

