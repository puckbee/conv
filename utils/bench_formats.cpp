#include "../bench/ncnn_conv_3x3.h"
#include <omp.h>

#ifdef _MKL_
#include <mkl_cblas.h>
#include <mkl.h>
#elif _OPENBLAS_
#include <cblas.h>
#endif

#include "bench_formats.h"
#include "mat.h"
#include "cpu.h"
#include "basic_operations.h"


inline float op_conv(Mat& blob, float* kernel, int start_w, int start_h, int kernel_size, int qqdan)
{
    float* in = (float*)blob;
    float sum = 0;
/*
    for(int kh=0; kh< kernel_size; kh++)
    {
        sum += in[(start_h+kh)*blob.w + start_w] * kernel[kh*kernel_size];
        sum += in[(start_h+kh)*blob.w + start_w+1] * kernel[kh*kernel_size + 1];
        sum += in[(start_h+kh)*blob.w + start_w+2] * kernel[kh*kernel_size + 2];
    }
*/




    for(int kh=0; kh< kernel_size; kh++)
    {
        for(int kw=0; kw<kernel_size; kw++)
        {
//            if(qqdan==1)
//            printf(" %f ", in[(start_h + kh)*blob.w + start_w + kw]);
            sum += in[(start_h + kh)*blob.w + start_w +kw] * kernel[kh * kernel_size + kw];
        }
//        if(qqdan==1)
//        printf("\n");
    }
/*
    for(int kh=0; kh< kernel_size; kh++)
    {
        for(int kw=0; kw<kernel_size; kw++)
        {
            if(qqdan==1)
            printf(" %f ", kernel[kh*kernel_size+kw]);
        }
        if(qqdan==1)
        printf("\n");
    }
            if(qqdan==1)
                printf(" sum=%f \n", sum);
*/
/*
    int x1 = start_h*blob.w + start_w;
    int x2 = x1 + blob.w + start_w;
    int x3 = x2 + blob.w + start_w;

    sum += in[x1] * kernel[0];
    sum += in[x1+1] * kernel[1];
    sum += in[x1+2] * kernel[2];
    
    sum += in[x2] * kernel[3];
    sum += in[x2+1] * kernel[4];
    sum += in[x2 +2] * kernel[5];

    sum += in[x3] * kernel[6];
    sum += in[x3 +1] * kernel[7];
    sum += in[x3 +2] * kernel[8];
*/


/*
    sum += in[(start_h)*blob.w + start_w] * kernel[0];
    sum += in[(start_h)*blob.w + start_w +1] * kernel[1];
    sum += in[(start_h)*blob.w + start_w +2] * kernel[2];
    
    sum += in[(start_h+1)*blob.w + start_w] * kernel[3];
    sum += in[(start_h+1)*blob.w + start_w +1] * kernel[4];
    sum += in[(start_h+1)*blob.w + start_w +2] * kernel[5];

    sum += in[(start_h+2)*blob.w + start_w] * kernel[6];
    sum += in[(start_h+2)*blob.w + start_w +1] * kernel[7];
    sum += in[(start_h+2)*blob.w + start_w +2] * kernel[8];
*/
    return sum;
}

int benchmark_naive_conv(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output )
{

    int inw = bottom_blob.w;
    int inh = bottom_blob.h;

    int kernel_w = kernel_size;
    int kernel_h = kernel_size;

    const int outw=(bottom_blob.w+ 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int outh=(bottom_blob.h+ 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;

    std::cout<<" outw = "<<outw<<"; outh = "<< outh<<std::endl;

    int inch = bottom_blob.c;
    int outch = num_output;

    top_blob.create(outw, outh, outch);
    memset(top_blob.data, 0, outw*outh*outch);


    Mat bottom_blob_bordered = bottom_blob;
    if(pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;
        inw = bottom_blob_bordered.w;
        inh = bottom_blob_bordered.h;
    }

    float* kernel_ptr = (float*) kernel_blob;

    float* top_ptr = (float*)top_blob;

#pragma omp parallel for
    for(int i=0; i< outch*outw*outh; i++)
    {
        top_ptr[i] = _bias_data.data[ i/(outw*outh)];
    }       
    
#pragma omp parallel for
    for(int p = 0; p < outch; p++)
    {
        float* out = top_blob.channel(p);
//        if(p==0)
//            printf(" begin .. out[0] = %f\n", out[0]);
        for(int q = 0; q < inch; q++)
        {
            Mat in = bottom_blob_bordered.channel(q);
            float* kernel= kernel_ptr + p* inch* kernel_size*kernel_size + q * kernel_size*kernel_size;

            for (int i_h = 0; i_h < outh; i_h++)
            {
                int start_h = i_h * stride_h;
                for(int i_w = 0; i_w < outw; i_w++)
                {
                    int start_w = i_w * stride_w;
                    int qqdan=0;
/*                    
                    if(i_h==0&&i_w==16&&p==0 &&q==0)
                    {
                        qqdan=1;

                    }
*/
                    out[i_h * outw + i_w] += op_conv(in, kernel, start_w,start_h,kernel_size, qqdan);
/*                 
                    int www9 = i_h * outw + i_w + p * outw * outh;
                    if((www9 < 2813 && www9 >= 2809))
                        printf("p = %d, i_h = %d, outw = %d, i_w = %d, out = %f \n", p, i_h, outw, i_w, out[i_h * outw + i_w]);
*/                        

//                    if(i_h==0&&i_w==16&&p==0 &&q==0)
//                        printf(" .... out[0] = %f\n", out[0]);

                    //std::cout<<" i_h="<<i_h<<"; i_w="<<i_w<<"; out="<<out[i_h*outw+i_w]<<endl;

                }

//                if(p==1)
//                   printMatrix("out", out, 1,5);
            }
            
//    printMatrix("top", top_blob.data + 2809, 1,5);
//    if(p==1)
//         printMatrix("out", out, 1,5);
        }
    }
}
int benchmark_ncnn_conv(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output )
{

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;


    const int stride = stride_w;

    Mat bottom_blob_bordered = bottom_blob;

    // padding the img if needed		
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_w == -233 && pad_h == -233)
    {
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

//    printBlob("Bottom", bottom_blob);	
//    printBlob("Bottom_bordered", bottom_blob_bordered);	

    int outw = (w - kernel_size) / stride + 1;
    int outh = (h - kernel_size) / stride + 1;

    std::cout<<" w,h = "<< outw <<", "<<outh<<endl;

    top_blob.create(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

	conv3x3s1_sse(bottom_blob_bordered, top_blob, kernel_blob, _bias_data);

//    printBlob("Top", top_blob);

	return 0;

}



void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda =  K ;
  int ldb =  N ;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}




int benchmark_im2col_gemm(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output )
{
	float* data_im;    //data of the bottom_blob in float* style 
	float* data_col;   //data after im2col transformation
	int kernel_h = kernel_size;
	int kernel_w = kernel_size;
    
    const int output_h = (bottom_blob.h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (bottom_blob.w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	mat2vector(bottom_blob, data_im);





    double t3 = microtime();
	im2col_cpu(data_im, bottom_blob.c, bottom_blob.h, bottom_blob.w,  kernel_h,  kernel_w,
     pad_h,  pad_w, stride_h,  stride_w, dilation_h,  dilation_w,
     data_col);
    double t4 = microtime();
    std::cout<<" im2col Time is "<< t4 - t3 <<endl;

    std::cout<<" w,h = "<< output_w <<", "<<output_h<<endl;
#ifdef _MKL_
	const CBLAS_LAYOUT Order=CblasRowMajor;
	const CBLAS_TRANSPOSE TransA=CblasNoTrans;
	const CBLAS_TRANSPOSE TransB=CblasNoTrans;
#elif _OPENBLAS_
	const enum CBLAS_ORDER Order=CblasRowMajor;
	const enum CBLAS_TRANSPOSE TransA=CblasNoTrans;
	const enum CBLAS_TRANSPOSE TransB=CblasNoTrans;
#endif
	const int M= num_output;//numRows of A; numCols of C
	const int N= output_h * output_w;//numCols of B and C
	const int K=kernel_w * kernel_h * bottom_blob.c;//numCols of A; numRows of B
	const float alpha=1;
	const float beta=0;
	const int lda=K;//numCols of A
	const int ldb=N;//numCols of B
	const int ldc=N;//numCols of C



    float* result = (float*) malloc (sizeof(float) * num_output * output_h * output_w);

    double t9 = microtime();
    #pragma omp parallel for
    for (int cc=0; cc< num_output*output_h*output_w; cc++)
    {
//        for (int ii=0; ii<N; ii++)
            result[cc] = _bias_data.data[cc%4];
    }
    double t10 = microtime();
    std::cout<<" _bias copy Time is "<< t10 - t9<<endl;


    const float* kernel = kernel_blob;

    double t5 = microtime();
//	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, kernel, lda, data_col, ldb, beta,  result, ldc);


	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_output, output_w * output_h, 9*bottom_blob.c, 1., kernel, data_col, 0., result);

/*
    for(int i=0; i< M; i++)
        for(int j=0; j<N; j++)
            for(int p=0; p<K; p++)
                result[i*N+j] += kernel[i*K+p] * data_col[p*N+j];
*/

    double t6 = microtime();
    std::cout<<" gemm Time is "<< t6 - t5<<endl;
    top_blob = Mat(output_w, output_h, num_output, result);






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
}









float* matrixTranspose(float* a, int m, int n)
{
   float* re = (float*) malloc( sizeof(float) * m * n);
   int i,j;
   for(i=0; i<m; i++)
     for(j=0; j<n; j++)
       re[j*m + i] = a [i*n+j];

   return re;
}

float* matrixMul(float* a, float* b, int m, int k, int n, float* re)
{
//   float* re = (float*) malloc( sizeof(float) * m * n);

   int i,j,l;

   for(i=0; i< m; i++)
     for(j=0; j<n; j++)
     {
        float sum = 0;
        for(l=0; l<k; l++)
          sum+= a[i*k+l] * b[l*n+j];
        re[i*n+j] = sum;
     }

   return re;
}

float* matrixDotProduct(float* a, float*b, int m, int n, float* re)
{
//   float* re = (float*) malloc(sizeof(float) * m * n);
 
   int i,j;

   for(i = 0; i<m; i++)
      for(j=0; j<n; j++)
        re[i*n+j] = a[i*n+j] * b[i*n+j];

   return re;
}


int winograd_m2r3(float* matrix_d, float* matrix_g, float* matrix_Gg, float* matrix_U, float* matrix_Btd, float* matrix_V, float* matrix_UV, float* matrix_AtUV, float* matrix_re)
{
 
   float matrix_Bt[16] = {1,0,-1,0,0,1,1,0,0,-1,1,0,0,1,0,-1};
//   float* matrix_B = matrixTranspose(matrix_Bt, 4,4);
   float matrix_B[16]  = {1,0,0,0,0,1,-1,1,-1,1,1,0,0,0,0,-1};

   float  matrix_G[12] = {1,0,0,0.5,0.5,0.5,0.5,-0.5,0.5,0,0,1};
//   float* matrix_Gt = matrixTranspose(matrix_G, 4,3);
   float matrix_Gt[12] = {1,0.5,0.5,0,0,0.5,-0.5,0,0,0.5,0.5,1};

   float matrix_At[8] = {1,1,1,0,0,1,-1,-1};
//   float* matrix_A = matrixTranspose(matrix_At, 2,4);
   float  matrix_A[8] = {1,0,1,1,1,-1,0,-1};

   matrixMul(matrix_G, matrix_g, 4,3,3, matrix_Gg);
   matrixMul(matrix_Gg, matrix_Gt, 4,3,4, matrix_U);

   matrixMul(matrix_Bt, matrix_d, 4,4,4, matrix_Btd);
   matrixMul(matrix_Btd, matrix_B, 4,4,4, matrix_V);

//   float* matrix_U = matrixMul(matrixMul(matrix_G, matrix_g, 4,3,3), matrix_Gt, 4,3,4);
//   float* matrix_V = matrixMul(matrixMul(matrix_Bt, matrix_d, 4,4,4), matrix_B, 4,4,4);

   printMatrix("UUUUU", matrix_U, 4,4);
//   printMatrix("VVVVV", matrix_V, 4,4);


   matrixDotProduct(matrix_U, matrix_V, 4,4, matrix_UV);
   matrixMul(matrix_At, matrix_UV, 2,4,4, matrix_AtUV);


   matrixMul(matrix_AtUV, matrix_A, 2,4,2, matrix_re);

//   float* re = matrixMul(matrixMul(matrix_At, matrix_UV, 2,4,4), matrix_A, 2,4,2);

//   return re;  
   return 0;
}

float* conv_direct(float* d, float* g, int d_h, int d_w)
{

   int g_w = 3;
   int g_h = 3;

   int slide = 1;
   
   int re_w = d_w - g_w + 1;
   int re_h = d_h - g_h + 1;

   float* re = (float*) malloc ( sizeof(float) * re_w * re_h);

   int i,j,k,l;

   for(i=0; i*slide + g_h <= d_h; i++)
      for(j=0; j*slide + g_w <= d_w; j++)
      {
         float sum = 0;
         for(k=0; k<g_h; k++)
            for(l=0; l<g_w; l++)
              sum += d[(i*slide+k) * d_w + j*slide + l ] * g[k*g_w + l];
         
         re[i*re_w + j] = sum;
//         if(i==5) printf(" sum = %f \n", sum);
      }

    return re;

}


//float* winograd(float* matrix_d, float* matrix_g, int d_h, int d_w)
int benchmark_naive_winograd(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& _bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output)
{

    float* matrix_Gg = (float*)malloc(sizeof(float) * 4 * 3);
    float* matrix_U = (float*)malloc(sizeof(float) * 4 * 4);
    float* matrix_Btd = (float*)malloc(sizeof(float) * 4 * 4);
    float* matrix_V = (float*)malloc(sizeof(float) * 4 * 4);

    float* matrix_UV = (float*)malloc(sizeof(float) * 4 * 4);
    float* matrix_AtUV = (float*)malloc(sizeof(float) * 2 * 4);
    float* matrix_re = (float*)malloc(sizeof(float) * 2  * 2);

    if(kernel_size != 3)
    {
        std::cout<<" Kernel Size is not 3"<<endl;
        return -1;
    }

    // initilize the variables
    int in_ch= bottom_blob.c;
    int in_w = bottom_blob.w;
    int in_h = bottom_blob.h;

    int out_ch = num_output;

    int wino_m = 2;
    int wino_r = 3;     // wino_r is equal to kernel_size
    int tile_w = wino_m + wino_r - 1;           // 4
    int tile_h = wino_m + wino_r - 1;           // 4

    int out_w = in_w - (wino_r - 1);
    int out_h = in_h - (wino_r - 1);

    float* winoTile = (float*) malloc(sizeof(float) * tile_w * tile_h);

//   int re_w = d_w - g_w + 1;  // width - (r-1)
//   int re_h = d_h - g_h + 1;
//   float* re = (float*) malloc(sizeof(float) * re_w * re_h);

//   int numTile_w = (d_w - tile_w) / (g_w - 1) + 1;
//   int numTile_h = (d_h - tile_h) / (g_h - 1) + 1;



    top_blob.create(out_w, out_h, out_ch);
 
   int i,j,k,l, s,t;

   int oc,ic;
   memset(top_blob.data, 0, out_w * out_h*out_ch * sizeof(float));
   float* top_ptr = (float*)top_blob;

#pragma omp parallel for
    for(int i=0; i< out_ch*out_w*out_h; i++)
    {
        top_ptr[i] = _bias_data.data[ i/(out_w*out_h)];
    }       


   for(oc = 0; oc < out_ch; oc ++)
   {
       float* refOut = (float*) top_blob.channel(oc);

//       if(oc%16 == 0)
//           std::cout<<" processing the "<<oc<<" channel of top_blob"<<endl;

       for(ic = 0; ic < in_ch; ic ++)
       {
           // wino_r eqs kernel_size
           float* kernel = (float*)kernel_blob + oc*in_ch*wino_r*wino_r + ic*wino_r*wino_r; 
           float* imgIn  = (float*)bottom_blob.channel(ic);
           for( i = 0; i < out_h; i+=wino_m)
               for ( j = 0; j < out_w; j+=wino_m)
               {
                   for(k=0; k< tile_h; k++)
                       for(l=0; l<tile_w; l++)
                           winoTile[k*tile_w + l] = imgIn[(i+k)*in_w + j+l];
/*                   
                   if(i==0 && j==16 && oc==0 && ic==0)
                   {
                       printMatrix("winoTile", winoTile, 4,4);
                       printMatrix("kernel", kernel, 3,3);
                   }
*/


                   winograd_m2r3(winoTile, kernel, matrix_Gg, matrix_U, matrix_Btd, matrix_V, matrix_UV, matrix_AtUV, matrix_re);
                   


  /*                 
                   if(i==0 && j==16 && oc==0)
                   {
                       printMatrix("matrix_re", matrix_re,2,2);
                       printf(" result[16] = %f\n", refOut[16]);
                   }
  */
                   for(s = 0; s < wino_m; s++)
                       for(t = 0; t< wino_m; t++)
                       {
                           refOut[(i+s)*out_w + j+t] += matrix_re[s * wino_m + t];
                       }

                   if(ic==(in_ch-1) && oc==0 && i==0 && j==0)
                   {
//                        printMatrix("Btd in niave wino", matrix_Btd, 4,4);
//                        printMatrix("Gg", matrix_Gg, 4,3);
//                        printMatrix("Kernel", kernel, 3,3);
//                        printMatrix("Gt", matrix_Gt, 3,4);
//                        printMatrix("U in niave wino", matrix_U, 4,4);
//                        printMatrix("V in niave wino", matrix_V, 4,4);
//                        printMatrix(" AtUV ", matrix_AtUV, 2,4);
//                        printMatrix(" UV ", matrix_UV, 4,4);
//                        printMatrix("matrix re", matrix_re, 2,2);
//                        printMatrix("re", refOut+i*out_w+j, 1,2);
//                        printf(" Distance = %d\n", (refOut-top_blob.data));

                   }




                   
//                   if(i==0 && j==16 && oc==0)
//                       printf("..... result[16] = %f\n", refOut[16]);
               }
       }
   }




}































