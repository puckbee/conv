#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <math.h>
#include <omp.h>

#include <immintrin.h>

#include <assert.h>



#ifdef _MKL_
#include <mkl_cblas.h>
#include <mkl.h>
#elif _OPENBLAS_
#include <cblas.h>
#endif

#include "utils/mat.h"
#include "utils/bench_formats.h"
#include "utils/basic_operations.h"

#include "bench/ncnn_conv_3x3.h"
#include "bench/ncnn_wino_3x3.h"

#include "bench/hpWino_3x3.h"

using namespace std;




int main(int argc, char** argv)
{

	Mat bottom_blob;   //data from the bottom level
	Mat kernel_blob;   //data of kernel
	Mat xtop_blob;     //the reference result of the top data
	
	Mat bias_data;    //bias data


	int kernel_size, num_output;

  double t1,t2;      // start and end time;
  int omp_threads;
  int gemm_threads;

	int pad_w = 0;
	int pad_h = 0;
	int stride_w = 1;
	int stride_h = 1;
	int dilation_h = 1;
	int dilation_w = 1;

//	read_data(bottom_blob, kernel_blob, xtop_blob, bias_data, kernel_size, num_output);

    int inw = 56;
    int inh = 56;
    int inch = 256;

    int outch = 256;

    kernel_size = 3;
    num_output = outch;


    generate_data(bottom_blob, kernel_blob, bias_data, inch, outch, inw, inh, kernel_size);

    std::cout<<" Convolution Information "<< std::endl;
    std::cout<<" Genral:                 "<< endl;
    std::cout<<"        input_channel =  "<< bottom_blob.c<<endl;
    std::cout<<"       output_channel =  "<< outch<<endl;
    std::cout<<"          kernel_size =  "<< kernel_size<<endl;
    std::cout<<" Bottom:                 "<< std::endl;
    std::cout<<"                width =  "<< bottom_blob.w<<endl;            // this width is numCols
    std::cout<<"               height =  "<< bottom_blob.h<<endl;            // this height is numRows
    std::cout<<"              channel =  "<< bottom_blob.c<<endl;
    std::cout<<endl;
    std::cout<<endl;



    // ***********    Start Naive-Conv, a naive implementation of conv from NCNN"<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"          Naive-conv        "<<endl;
    std::cout<<"        *************      "<<endl;

    omp_threads = omp_get_max_threads();
    
    Mat top_naive_conv_blob;      //data result after naive convolution

    t1 = microtime();
    benchmark_naive_conv(bottom_blob, kernel_blob, top_naive_conv_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();

    // take the ncnn-conv for standard
    xtop_blob = top_naive_conv_blob;
    std::cout<<" Performance NCNN-conv "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;
//    printBlob("naive conv", top_naive_conv_blob);
    
//    printf(" w =%d, h = %d, cstep = %d \n", bottom_blob.w, bottom_blob.h, bottom_blob.cstep);

    // ***********    Start NCNN-Conv, a NCNN implementation of conv from NCNN"<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"          NCNN-conv        "<<endl;
    std::cout<<"        *************      "<<endl;

    omp_threads = omp_get_max_threads();
	Mat top_ncnn_conv_blob;      //data result after convolution
    t1 = microtime();
	benchmark_ncnn_conv(bottom_blob, kernel_blob, top_ncnn_conv_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();
	checkResults(top_ncnn_conv_blob.data, xtop_blob.data, top_ncnn_conv_blob.total());
//    checkResults(top_ncnn_conv_blob, xtop_blob);

	
    std::cout<<" Performance NCNN-conv "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;
//    printBlob("ncnn", top_ncnn_conv_blob);

    // ***********    Start im2col-GEMM, implementation with gemm after im2col"<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"         im2col-GEMM       "<<endl;
    std::cout<<"        *************      "<<endl;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads(omp_threads);
#ifdef _MKL_
    gemm_threads = mkl_get_max_threads();
//    gemm_threads = omp_threads;
#elif _OPENBLAS_
    gemm_threads = omp_threads;
#endif

	Mat top_gemm_blob;      //data result after convolution
    t1 = microtime();
    benchmark_im2col_gemm(bottom_blob, kernel_blob, top_gemm_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();
	checkResults(top_gemm_blob.data, xtop_blob);

    std::cout<<" Performance im2col-GEMM "<< gemm_threads <<" "<< t2 - t1 <<endl<<endl;
//    printBlob("gemm", top_gemm_blob);


    // ***********    Start ncnn-wino, implementation with gemm after im2col"<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"          ncnn-wino        "<<endl;
    std::cout<<"        *************      "<<endl;

    Mat top_nwino_blob(xtop_blob.w, xtop_blob.h, outch);

    // Note that the result is right only when we turn off the omp in these two functions.
    // But here we use the with-openmp version


    t1 = microtime();
		Mat kernel_tm(64*inch*outch);
		conv3x3s1_winograd64_transform_kernel_neon(kernel_blob, kernel_tm, inch, outch);

		conv3x3s1_winograd64_neon(bottom_blob, top_nwino_blob,kernel_tm,0);
    t2 = microtime();

//    checkResults(top_nwino_blob.data, xtop_blob.data, top_nwino_blob.total());
//    checkResults(top_nwino_blob, xtop_blob);
//    checkResults(top_nwino_blob.data, xtop_blob);
    std::cout<<" Performance ncnn-wino "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;

/*
    // ***********    Start serial-naive-wino. I simply wrote a serial naive winograd. "<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"      serial-naive-wino    "<<endl;
    std::cout<<"        *************      "<<endl;

    Mat top_naive_wino_blob;

    t1 = microtime();
    benchmark_naive_winograd(bottom_blob, kernel_blob, top_naive_wino_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();

    checkResults(top_naive_wino_blob.data, xtop_blob.data, top_naive_wino_blob.total());
    std::cout<<" Performance serial-naive-wino "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;
*/

    // ***********    My high performance Winograd. "<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"             hpWino        "<<endl;
    std::cout<<"        *************      "<<endl;

    Mat top_hp_wino_Naive_blob;

    t1 = microtime();
    benchmark_hp_winograd_Naive(bottom_blob, kernel_blob, top_hp_wino_Naive_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();

//    checkResults(top_hp_wino_Naive_blob.data, xtop_blob.data, top_hp_wino_Naive_blob.total());
    checkResultsAlign(top_hp_wino_Naive_blob.data, xtop_blob);
    std::cout<<" Performance hpwino "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;



    // ***********    My high performance Winograd. "<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"          hpWino-T         "<<endl;
    std::cout<<"        *************      "<<endl;

    Mat top_hp_wino_T_blob;

    t1 = microtime();
    benchmark_hp_winograd_T(bottom_blob, kernel_blob, top_hp_wino_T_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();

//    checkResultsAlign(top_hp_wino_T_blob.data, xtop_blob);
    std::cout<<" Performance hpwin-T "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;


    // ***********    My high performance Winograd. "<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"          hpWino-A         "<<endl;
    std::cout<<"        *************      "<<endl;

    Mat top_hp_wino_A_blob;

    t1 = microtime();
    benchmark_hp_winograd_A(bottom_blob, kernel_blob, top_hp_wino_A_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();

//    checkResults(top_hp_wino_A_blob.data, xtop_blob.data, top_hp_wino_A_blob.total());
    checkResultsAlign(top_hp_wino_A_blob.data, xtop_blob);
    std::cout<<" Performance hpwin-A "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;


    // ***********    My high performance Winograd. "<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"          hpWino-B         "<<endl;
    std::cout<<"        *************      "<<endl;

    Mat top_hp_wino_B_blob;

    t1 = microtime();
    benchmark_hp_winograd_B(bottom_blob, kernel_blob, top_hp_wino_B_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();

    checkResultsAlign(top_hp_wino_B_blob.data, xtop_blob);
    std::cout<<" Performance hpwin-B "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;


    // ***********    My high performance Winograd. "<<endl;
    std::cout<<" ************************* "<<endl;
    std::cout<<"          hpWino-C         "<<endl;
    std::cout<<"        *************      "<<endl;

    Mat top_hp_wino_C_blob;

    t1 = microtime();
    benchmark_hp_winograd_C(bottom_blob, kernel_blob, top_hp_wino_C_blob, bias_data, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, kernel_size, num_output);
    t2 = microtime();

    checkResultsAlign(top_hp_wino_C_blob.data, xtop_blob);
    std::cout<<" Performance hpwin-C "<< omp_threads <<" "<< t2 - t1 <<endl<<endl;








}

