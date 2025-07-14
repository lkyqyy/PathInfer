#include "postprocess.h"
#include <cmath>


inline float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

void decodeOutputs(const float* prob, int output_h, int output_w, float conf_thresh,float nms_thresh,
                   float scale, int pad_w, int pad_h, int img_w, int img_h,
                   std::vector<DetectResult>& results) 
{
    results.clear();
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    int num_classes = 2;
    int total_len = output_h * output_w;

    cv::Mat dout(output_h, output_w, CV_32F, (void*)prob);  
    cv::Mat det_output = dout.t();  // shape: (num_boxes, 4 + num_classes)

    for (int i = 0; i < det_output.rows; ++i) {
        float cx = det_output.at<float>(i, 0);
        float cy = det_output.at<float>(i, 1);
        float ow = det_output.at<float>(i, 2);
        float oh = det_output.at<float>(i, 3);

        cv::Mat scores = det_output.row(i).colRange(4, 4 + num_classes).clone();
        for (int j = 0; j < num_classes; ++j)
            scores.at<float>(0, j) = 1.f / (1.f + expf(-scores.at<float>(0, j)));  // sigmoid

        cv::Point classIdPoint;
        double max_score;
        cv::minMaxLoc(scores, 0, &max_score, 0, &classIdPoint);
        

        if (max_score > 0.6) {
            int x = static_cast<int>(((cx - 0.5f * ow) - pad_w) / scale);
            int y = static_cast<int>(((cy - 0.5f * oh) - pad_h) / scale);
            int w = static_cast<int>(ow / scale);
            int h = static_cast<int>(oh / scale);
            cv::Rect box = cv::Rect(cv::Point(x, y), cv::Size(w, h)) & cv::Rect(0, 0, img_w, img_h);

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(static_cast<float>(max_score));
        }
    }
    std::vector<int> indices;
    //cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, nms_thresh, indices);

    nms_cuda(boxes, confidences, 0.6, indices);//Debug 时候发现没有把正确的参数传递进来，所以设置了0.6


    for (int idx : indices) {
        DetectResult dr;
        dr.box = boxes[idx];
        dr.classId = classIds[idx];
        dr.conf = confidences[idx];
        results.push_back(dr);
    }
}
