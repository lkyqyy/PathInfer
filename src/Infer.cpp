#include "Infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

using namespace nvinfer1;


class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// Impl定义
struct IHC_TRT::Impl {
    float conf_thresh=0.6f;
    float nms_thresh=0.5 ;
    int input_h;
    int input_w;
    int output_h;
    int output_w;
    int max_batch = 8;  

    IRuntime* runtime{ nullptr };
    ICudaEngine* engine{ nullptr };
    IExecutionContext* context{ nullptr };
    
    void* buffers[2]{ nullptr, nullptr };  

    std::vector<float> prob;             
    cudaStream_t stream;
};


IHC_TRT::IHC_TRT() : pimpl_(new Impl()) {}

IHC_TRT::~IHC_TRT()
{
    if (pimpl_->stream) {
        cudaStreamSynchronize(pimpl_->stream);
        cudaStreamDestroy(pimpl_->stream);
    }
    if (pimpl_->buffers[0]) cudaFree(pimpl_->buffers[0]);
    if (pimpl_->buffers[1]) cudaFree(pimpl_->buffers[1]);
    if (pimpl_->context) pimpl_->context->destroy();
    if (pimpl_->engine) pimpl_->engine->destroy();
    if (pimpl_->runtime) pimpl_->runtime->destroy();
    delete pimpl_;

}

bool IHC_TRT::Init(const std::string& engine_path, float conf_thresh, float nms_thresh) 
{
    pimpl_->conf_thresh = conf_thresh;
    pimpl_->nms_thresh = nms_thresh;

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) 
    {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    pimpl_->runtime = createInferRuntime(gLogger);
    pimpl_->engine = pimpl_->runtime->deserializeCudaEngine(engineData.data(), size);
    pimpl_->context = pimpl_->engine->createExecutionContext();

    int input_index = pimpl_->engine->getBindingIndex("images");
    int output_index = pimpl_->engine->getBindingIndex("output0");

    const Dims input_dims = pimpl_->engine->getBindingDimensions(input_index);


 

    pimpl_->input_h = pimpl_->engine->getBindingDimensions(input_index).d[2];
    pimpl_->input_w = pimpl_->engine->getBindingDimensions(input_index).d[3];

    const Dims output_dims = pimpl_->engine->getBindingDimensions(output_index);
    pimpl_->output_h = pimpl_->engine->getBindingDimensions(output_index).d[1];
    pimpl_->output_w = pimpl_->engine->getBindingDimensions(output_index).d[2];

    pimpl_->max_batch = 8;

    // 输入输出缓冲区要按最大 batch 分配
    size_t input_size = pimpl_->max_batch * 3 * pimpl_->input_h * pimpl_->input_w * sizeof(float);
    size_t output_size = pimpl_->max_batch * pimpl_->output_h * pimpl_->output_w * sizeof(float);

    cudaMalloc(&pimpl_->buffers[input_index], input_size);
    cudaMalloc(&pimpl_->buffers[output_index], output_size);

    pimpl_->prob.resize(pimpl_->max_batch * pimpl_->output_h * pimpl_->output_w);
    cudaStreamCreate(&pimpl_->stream);

    return true;
}

bool IHC_TRT::InferenceBatch(const std::vector<cv::Mat>& images, std::vector<std::vector<DetectResult>>& batch_results) 
{
    int batch_size = static_cast<int>(images.size());
    if (batch_size == 0) return false;

    nvinfer1::Dims input_dims = pimpl_->engine->getBindingDimensions(0);
    input_dims.d[0] = batch_size;
    pimpl_->context->setBindingDimensions(0, input_dims);


    size_t input_size = batch_size * 3 * pimpl_->input_h * pimpl_->input_w * sizeof(float);
    size_t output_size = batch_size * pimpl_->output_h * pimpl_->output_w * sizeof(float);

    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    //resize + normalize + CHW + padding
    std::vector<float> scales(batch_size);
    std::vector<int> pads_w(batch_size), pads_h(batch_size);
    preprocessBatchCUDA(images, static_cast<float*>(d_input), batch_size, pimpl_->input_w, 
                        pimpl_->input_h, pimpl_->stream, scales, pads_w, pads_h);

    void* bindings[2] = { d_input, d_output };
    pimpl_->context->enqueueV2(bindings, pimpl_->stream, nullptr);


    std::vector<float> output_host(output_size / sizeof(float));
    cudaMemcpyAsync(output_host.data(), d_output, output_size, cudaMemcpyDeviceToHost, pimpl_->stream);
    cudaStreamSynchronize(pimpl_->stream);


    batch_results.clear();
    for (int i = 0; i < batch_size; ++i) 
    {
        std::vector<DetectResult> results;
        const float* ptr = output_host.data() + i * pimpl_->output_h * pimpl_->output_w;

        decodeOutputs(ptr, pimpl_->output_h, pimpl_->output_w,
                      pimpl_->conf_thresh, pimpl_->nms_thresh, scales[i], pads_w[i], pads_h[i],
                      images[i].cols, images[i].rows, results);

        batch_results.push_back(std::move(results));
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return true;

}
