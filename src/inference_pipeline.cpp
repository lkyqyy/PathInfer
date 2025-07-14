#include "inference_pipeline.h"
#include "utils.h"
#include <condition_variable>
#include <queue>
#include <mutex>
#include <atomic>
#include "detect_result.h"


std::queue<PatchData> patch_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
bool producer_done = false;



// ------------------ producer_from_png (CV 图像) ------------------
void producer_from_png(const cv::Mat& full_image, const std::vector<cv::Point>& offsets,
                       int patch_size, std::atomic<int>& skipped_patch_count) {
    for (const auto& pt : offsets) {
        if (pt.x + patch_size > full_image.cols || pt.y + patch_size > full_image.rows) {
            skipped_patch_count++;
            continue;
        }

        cv::Mat patch = full_image(cv::Rect(pt.x, pt.y, patch_size, patch_size)).clone();
        if (patch.empty() || isBlankOrBlackPatch(patch)) {
            skipped_patch_count++;
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            patch_queue.push({patch, pt});
        }
        queue_cv.notify_one();
    }

    producer_done = true;
    queue_cv.notify_all();
}

// ------------------ consumer ------------------
void consumer(IHC_TRT& model,
              std::vector<DetectResult>& global_results,
              std::vector<double>& infer_times) {
    while (true) {
        std::vector<cv::Mat> batch_images;
        std::vector<cv::Point> batch_offsets;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [] {
                return patch_queue.size() >= 8 || (producer_done && !patch_queue.empty());
            });

            while (!patch_queue.empty() && batch_images.size() < 8) {
                PatchData item = patch_queue.front();
                patch_queue.pop();
                batch_images.push_back(item.image);
                batch_offsets.push_back(item.offset);
            }
        }

        if (!batch_images.empty()) {
            std::vector<std::vector<DetectResult>> results;
            auto t1 = std::chrono::high_resolution_clock::now();
            model.InferenceBatch(batch_images, results);
            auto t2 = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            infer_times.push_back(time_ms);

            for (size_t j = 0; j < results.size(); ++j) {
                if (results[j].empty()) continue;

                std::vector<DetectResult> filtered;
                for (const auto& det : results[j]) {
                    if (det.conf > 0.1f && det.box.area() > 0)
                        filtered.push_back(det);
                }

                mergePatchResults(filtered, batch_offsets[j], global_results);
            }
        }

        if (producer_done && patch_queue.empty()) break;
    }
}
