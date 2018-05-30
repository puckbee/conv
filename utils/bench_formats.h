#include "mat.h"
#include "cpu.h"

int benchmark_naive_conv(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output );
inline float op_conv(Mat& blob, float* kernel, int start_w, int start_h, int kernel_size);

int benchmark_ncnn_conv(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output );

int benchmark_im2col_gemm(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output );

void mat2vector(Mat& blob, float*& data_im);

void im2col_cpu(const float* data_im, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, float*& data_colx);

int benchmark_naive_winograd(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output);

