#include "preprocess.h"


cv::Mat letterboxResize(const cv::Mat& image, int target_w, int target_h, float& scale, int& pad_w, int& pad_h) 
{
    int src_w = image.cols;
    int src_h = image.rows;
    scale = std::min((float)target_w / src_w, (float)target_h / src_h);

    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    pad_w = (target_w - new_w) / 2;
    pad_h = (target_h - new_h) / 2;

    int top = pad_h, bottom = target_h - new_h - pad_h;
    int left = pad_w, right = target_w - new_w - pad_w;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

//  BGR → 转 RGB + 归一化 + NCHW
cv::Mat convertToBlob(const cv::Mat& image, int input_w, int input_h) 
{
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false); 
    return blob;
}

