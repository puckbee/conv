#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <mkl_cblas.h>

#include "mat.h"
#include "convolution_3x3.h"

using namespace std;
using namespace ncnn;




int read_data(Mat& bottom_blob, Mat& kernel_blob, Mat& xtop_blob, Mat& _bias_data, int &kernel_size, int& num_output)
{
	
	fstream fd;
	fd.open("1.txt");


	string strLine;
	getline(fd,strLine);
	
	sscanf(strLine.c_str(), " %d %d %d, %d %d, %d %d %d\n", &bottom_blob.w, &bottom_blob.h, &bottom_blob.c, &kernel_size, &num_output, &xtop_blob.w, &xtop_blob.h, &xtop_blob.c);

	printf("bottom_w =  %d, bottom_h =  %d, bottom_c =  %d; kernel_width = %d, kernel_height = %d, kernel_c = %d; xtop_w = %d, xtop_h = %d, xtop_c = %d\n", bottom_blob.w, bottom_blob.h, bottom_blob.c, kernel_size, kernel_size, num_output, xtop_blob.w, xtop_blob.h, xtop_blob.c);

	bottom_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.c);
//	kernel_blob.create(kernel_blob.w, kernel_blob.h, kernel_blob.c);
	kernel_blob.create(kernel_size * kernel_size * bottom_blob.c * num_output);
	xtop_blob.create(xtop_blob.w, xtop_blob.h, xtop_blob.c);
//	bottom_blob.data = (float*) malloc(sizeof(float) * bottom_blob.w * bottom_blob.h * bottom_blob.c);
//	kernel_blob.data = (float*) malloc(sizeof(float) * kernel_blob.w * kernel_blob.h * kernel_blob.c);
//  	   xtop_blob.data = (float*) malloc(sizeof(float) * xtop_blob.w * xtop_blob.h * xtop_blob.c);
	
//	_bias_data.data = (float*)malloc (sizeof(float) * xtop_blob.c);

//	_bias_data.create(1,1,xtop_blob.c);
	_bias_data.create(num_output);

	if(bottom_blob.data == NULL)
	{
		printf(" bottom is null");
		return -1;
	}
	if(kernel_blob.data == NULL)
	{
		printf(" kernel is null");
		return -1;
	}
	if(xtop_blob.data == NULL)
	{
		printf(" xtop is null");
		return -1;
	}
	
    std::cout<<"bottom:"<<endl;
	int p= 0;
	getline(fd,strLine);
	istringstream iss(strLine);
	float xxx;
	while(iss)
	{
		if(p>= bottom_blob.w*bottom_blob.h*bottom_blob.c)
		   break;
		iss>>xxx;
//		std::cout<<xxx<<" ";
		int cc = p/(bottom_blob.w * bottom_blob.h);
		int seg = p%(bottom_blob.w * bottom_blob.h);
		
		bottom_blob.channel(cc)[seg] = xxx;

		p++;
//		printf("%f cc=%d, seg=%d; ",xxx, cc, seg);
		printf("%f ",xxx);
	}
	std::cout<<endl;
    std::cout<<"kernel:"<<endl;
	p = 0;
	getline(fd,strLine);
	getline(fd,strLine);
	iss.clear();
	iss.rdbuf()->str(strLine);
	iss.seekg(0,ios::beg);
	while(iss)
	{
		if(p>= kernel_size*kernel_size*bottom_blob.c * num_output)
		   break;
		iss>>xxx;
//		std::cout<<xxx<<" ";
		printf("%f ",xxx);
//		int cc = p/(kernel_blob.w * kernel_blob.h);
//		int seg = p%(kernel_blob.w * kernel_blob.h);
//		kernel_blob.channel(cc)[seg] = xxx;
		kernel_blob[p] = xxx;
		p++;
	}
	std::cout<<endl;
    std::cout<<"bias:"<<endl;
	p = 0;
	getline(fd,strLine);
	iss.clear();
	iss.rdbuf()->str(strLine);
	iss.seekg(0,ios::beg);
	while(iss)
	{
		if(p>= _bias_data.w*_bias_data.h*_bias_data.c)
		   break;
		iss>>xxx;
//		std::cout<<xxx<<" ";
		printf("%f ",xxx);
//		int cc = p/(_bias_data.w * _bias_data.h);
//		int seg = p%(_bias_data.w * _bias_data.h);
//		_bias_data.channel(cc)[seg] = xxx;
		_bias_data[p] = xxx;
		p++;
	}
    std::cout<<endl;
    std::cout<<"top_blob:"<<endl;
	p = 0;
	getline(fd,strLine);
	getline(fd,strLine);
	iss.clear();
	iss.rdbuf()->str(strLine);
	iss.seekg(0,ios::beg);
	while(iss)
	{
		if(p>= xtop_blob.w*xtop_blob.h*xtop_blob.c)
		   break;
		iss>>xxx;
//		std::cout<<xxx<<" ";
		printf("%f ",xxx);
		int cc = p/(xtop_blob.w * xtop_blob.h);
		int seg = p%(xtop_blob.w * xtop_blob.h);
		xtop_blob.channel(cc)[seg] = xxx;
		p++;
	}
	std::cout<<endl;

	return 0;
}

int ncnn_conv(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int kernel_size, int num_output )
{

	printf(" enter in this function\n");
    int w = bottom_blob.w;
    int h = bottom_blob.h;
	int inch = bottom_blob.c;


//    const int kernel_size = kernel_blob.w;
    const int stride = stride_w;
//	const int num_output = kernel_blob.c;

    Mat bottom_blob_bordered = bottom_blob;
		
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f);
		printf(" 1\n");
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_w == -233 && pad_h == -233)
    {
		printf(" 2\n");
        int wpad = kernel_size + (w - 1) / stride * stride - w;
        int hpad = kernel_size + (h - 1) / stride * stride - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

	printf(" conv starts\n");
	int nnz = 0;

	for(int cc=0; cc<bottom_blob.c; cc++)
	{	
//		printf("\nChannel %d:\n",cc);
		for(int ii=0; ii<bottom_blob.h; ii++)
			for(int jj=0; jj<bottom_blob.w; jj++)
			{
	//			printf("%f cc=%d, seg=%d; ", (float)bottom_blob.channel(cc)[ii*bottom_blob.w + jj], cc, ii*bottom_blob.w+jj);
				printf("%f ", (float)bottom_blob.channel(cc)[ii*bottom_blob.w + jj]);
				if(bottom_blob.channel(cc)[ii*bottom_blob.w+jj] != 0)
					nnz++;
			}
	}
	
	printf("\n");

	for(int cc=0; cc<bottom_blob_bordered.c; cc++)
	{	
//		printf("\nChannel %d:\n",cc);
		for(int ii=0; ii<bottom_blob_bordered.h; ii++)
			for(int jj=0; jj<bottom_blob_bordered.w; jj++)
			{
	//			printf("%f cc=%d, seg=%d; ", (float)bottom_blob_bordered.channel(cc)[ii*bottom_blob_bordered.w + jj], cc, ii*bottom_blob_bordered.w+jj);
				printf("%f ", (float)bottom_blob_bordered.channel(cc)[ii*bottom_blob_bordered.w + jj]);
				if(bottom_blob_bordered.channel(cc)[ii*bottom_blob_bordered.w+jj] != 0)
					nnz++;
			}
	}

	printf("\n");

    int outw = (w - kernel_size) / stride + 1;
    int outh = (h - kernel_size) / stride + 1;

    top_blob.create(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

//	printf(" outw = %d, outh = %d, num_output = %d \n", outw, outh, num_output);
	conv3x3s1_sse(bottom_blob_bordered, top_blob, kernel_blob, _bias_data);

	printf(" after conv top_blob.c=%d num_output=%d\n", top_blob.c, num_output);
//	printf(" ncnn_tops: w=%d, h=%d, c=%d \n", top_blob.w, top_blob.h, top_blob.c);

	for(int cc=0; cc<top_blob.c; cc++)
	{	
//		printf("\nChannel %d:\n",cc);
		for(int ii=0; ii<top_blob.h; ii++)
			for(int jj=0; jj<top_blob.w; jj++)
			{
				printf("%f ", (float)top_blob.channel(cc)[ii*top_blob.w + jj]);
				if(top_blob.channel(cc)[ii*top_blob.w+jj] != 0)
					nnz++;
			}
	}

	return 0;

}

void mat2vector(Mat& blob, float*& data_im)
{
	data_im = (float*) malloc( sizeof(float) * blob.w * blob.h * blob.c);
	int im=0;
	for(int cc=0; cc<blob.c; cc++)
		for(int ii = 0; ii<blob.h; ii++)
			for(int jj=0; jj<blob.w; jj++)
				data_im[im++] = blob.channel(cc)[ii*blob.w + jj];
}


void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float*& data_colx) {
  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  data_colx = (float*) malloc(sizeof(float) * output_h * output_w * channels * kernel_h * kernel_w);
  float* data_col = data_colx;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
  printf("\nim2col: \n");
  for (int cc = 0; cc < channels*kernel_h*kernel_w; cc++)
//   for (int ii = 0; ii< output_h * output_w; ii++)
//	  printf("%f ", data_colx[cc * output_h * output_w + ii]);
	  printf("%f ", data_colx[cc * output_h * output_w]);

  printf(" \n");
}



int main(int argc, char** argv)
{
	
	Mat bottom_blob;
	Mat kernel_blob;
	Mat top_blob;
	Mat xtop_blob;
	
	Mat _bias_data;

	float* data_im;
	float* data_col;


	int pad_w = 1;
	int pad_h = 1;
	int stride_w = 1;
	int stride_h = 1;
	int dilation_h = 1;
	int dilation_w = 1;


	int kernel_size, num_output;

	read_data(bottom_blob, kernel_blob, xtop_blob, _bias_data, kernel_size, num_output);

	ncnn_conv(bottom_blob, kernel_blob, top_blob, _bias_data, pad_w, pad_h, stride_w, stride_h, kernel_size, num_output);

	mat2vector(bottom_blob, data_im);

	int kernel_h = kernel_size;
	int kernel_w = kernel_size;

	im2col_cpu(data_im, bottom_blob.c, bottom_blob.h, bottom_blob.w,  kernel_h,  kernel_w,
     pad_h,  pad_w, stride_h,  stride_w, dilation_h,  dilation_w,
     data_col);

  const int output_h = (bottom_blob.h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (bottom_blob.w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    printf("\nim2col ... 2: \n");
    for (int cc = 0; cc < bottom_blob.c*kernel_h*kernel_w; cc++)
//   for (int ii = 0; ii< output_h * output_w; ii++)
	  printf("%f ", data_col[cc * output_h * output_w]);

    printf("\n");


	const CBLAS_LAYOUT Order=CblasRowMajor;
	const CBLAS_TRANSPOSE TransA=CblasNoTrans;
	const CBLAS_TRANSPOSE TransB=CblasNoTrans;
	const int M= top_blob.c;//A的行数，C的行数
	const int N= output_h * output_w;//B的列数，C的列数
	const int K=kernel_w * kernel_h * bottom_blob.c;//A的列数，B的行数
	const float alpha=1;
	const float beta=1;
	const int lda=K;//A的列
	const int ldb=N;//B的列
	const int ldc=N;//C的列
	
    float* result = (float*) malloc (sizeof(float) * K * N * top_blob.c);

    for (int cc=0; cc< top_blob.c; cc++)
    {
        for (int ii=0; ii<K*N; ii++)
            result[cc*K*N + ii] = _bias_data.data[cc];
    }

    const float* kernel = kernel_blob;

    float sum = 0;
    for(int ii=0; ii< K; ii++)
    {
        sum += kernel[ii] * data_col[ii*N];
        printf(" %f x %f ;", data_col[ii*N], kernel[ii]);
    }

    printf("\n");
    printf(" sum = %f\n", sum );

	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, kernel, lda, data_col, ldb, beta,  result, ldc);

	printf(" GEMM \n");
	for(int cc=0; cc< top_blob.c; cc++)
		for ( int ii = 0; ii < top_blob.w * top_blob.h; ii++)
			printf("%f ", result[cc * top_blob.w * top_blob.h + ii]);
	printf(" \n");

//	delete(bottom_blob.data);
//	delete(kernel_blob.data);
//	delete(top_blob.data);
//	delete(xtop_blob.data);

}




