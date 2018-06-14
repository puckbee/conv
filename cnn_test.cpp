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

using namespace std;

//typedef __attribute__((aligned(64))) union zmmd {                     
typedef union zmmd {                     
        __m256d regpd;        
        __m256i regi32;      
        __m256 regps;      
        double elemspd[4]; 
        int elemsi32[8];
        float elemsps[8];
} zmmd_t;



class Wino
{
public:
    float* Bt;     // t x t
    float* B;      // t x t
    float* G;      // t x r
    float* Gt;     // r x t
    float* At;     // m x t
    float* A;      // t x m

    int sw = 8;    // simd width

    float* U;
    float* V;

    float* U_A;
    float* V_A;

    float* U_B;
    float* V_B;

    float* U_C;
    float* V_C;

    float* UV_A;
    float* UV_B;
    float* UV_C;

    float* tileR;

    int m;
    int r;
    int t;

    int in_w;
    int in_h;
    int in_ch;

    int out_w;
    int out_h;
    int out_ch;

    int kernel_size;

    int numTiles;
    int numTiles_w;
    int numTiles_h;

    Wino(int _m, int _r);
    void createU(Mat& kernel_blob);
    void createV(Mat& bottom_blob);
    void getNaiveResult(Mat& top_blob);

    void convertU_A();
    void convertV_A();
    void getResult_A(Mat& top_blob);

    void convertU_B();
    void convertV_B();
    void getResult_B(Mat& top_blob);

    void convertU_C();
    void convertV_C();
    void getResult_C(Mat& top_blob);


    ~Wino();

};

Wino::Wino(int _m, int _r)
{
    m = _m; 
    r = _r; 
    t = m + r - 1;

    if(m == 2 && r == 3)
    {
       float matrix_Bt[16] = {1,0,-1,0,0,1,1,0,0,-1,1,0,0,1,0,-1};
       Bt = (float*)_mm_malloc(sizeof(float)*16, 32);
       memcpy(Bt, matrix_Bt, 16*sizeof(float));

       float matrix_B[16]  = {1,0,0,0,0,1,-1,1,-1,1,1,0,0,0,0,-1};
       B = (float*)_mm_malloc(sizeof(float)*16, 32);
       memcpy(B, matrix_B, 16*sizeof(float));

       float  matrix_G[12] = {1,0,0,0.5,0.5,0.5,0.5,-0.5,0.5,0,0,1};
       G = (float*)_mm_malloc(sizeof(float)*12, 32);
       memcpy(G, matrix_G, 12*sizeof(float));

       float matrix_Gt[12] = {1,0.5,0.5,0,0,0.5,-0.5,0,0,0.5,0.5,1};
       Gt = (float*)_mm_malloc(sizeof(float)*12, 32);
       memcpy(Gt, matrix_Gt, 12*sizeof(float));

       float matrix_At[8] = {1,1,1,0,0,1,-1,-1};
       At = (float*)_mm_malloc(sizeof(float)*8, 32);
       memcpy(At, matrix_At, 8*sizeof(float));

       float  matrix_A[8] = {1,0,1,1,1,-1,0,-1};
       A = (float*)_mm_malloc(sizeof(float)*8, 32);
       memcpy(A, matrix_A, 8*sizeof(float));
    }

    if(m == 4 && r == 3)
    {
       float matrix_Bt[36] = {4,0,-5,0,1,0, 0,-4,-4,1,1,0, 0,4,-4,-1,1,0, 0,-2,-1,2,1,0, 0,2,-1,-2,1,0, 0,4,0,-5,0,1};
       Bt = (float*)_mm_malloc(sizeof(float)*36, 32);
       memcpy(Bt, matrix_Bt, 36*sizeof(float));

       float matrix_B[36]  = {4,0,0,0,0,0, 0,-4,4,-2,2,4, -5,-4,-4,-1,-1,0, 0,1,-1,2,-2,-5, 1,1,1,1,1,0, 0,0,0,0,0,1};
       B = (float*)_mm_malloc(sizeof(float)*36, 32);
       memcpy(B, matrix_B, 36*sizeof(float));

       float  matrix_G[18] = {0.25,0,0, -0.166666666,-0.166666666,-0.166666666, -0.166666666,0.166666666,-0.166666666, 0.041666666,0.08333333,0.166666666, 0.04166666666,-0.083333333,0.166666666, 0,0,1};
 /*
       float  matrix_G[18] = {   0.25,       0,       0,
                              -1.0f/6, -1.0f/6, -1.0f/6,
                              -1.0f/6,  1.0f/6, -1.0f/6,
                              1.0f/24, 1.0f/12,  1.0f/6,
                              1.0f/24,-1.0f/12,  1.0f/6,
                                    0,       0,       1
                             };
*/                             
       G = (float*)_mm_malloc(sizeof(float)*18, 32);
       memcpy(G, matrix_G, 18*sizeof(float));

       float matrix_Gt[18] = {0.25,-0.166666666,-0.166666666,0.041666666,0.041666666,0, 0,-0.166666666,0.166666666,0.08333333,-0.08333333,0, 0,-0.166666666,-0.166666666,0.166666666,0.166666666,1};
/*       
       float matrix_Gt[18] = {  0.25, -1.0f/6, -1.0f/6, 1.0f/24, 1.0f/24, 0,
                                   0, -1.0f/6,  1.0f/6, 1.0f/12,-1.0f/12, 0,
                                   0, -1.0f/6, -1.0f/6,  1.0f/6,  1.0f/6, 1 
                             };
*/                             
       Gt = (float*)_mm_malloc(sizeof(float)*18, 32);
       memcpy(Gt, matrix_Gt, 18*sizeof(float));

       float matrix_At[24] = {1,1,1,1,1,0, 0,1,-1,2,-2,0, 0,1,1,4,4,0, 0,1,-1,8,-8,1};
       At = (float*)_mm_malloc(sizeof(float)*24, 32);
       memcpy(At, matrix_At, 24*sizeof(float));

       float  matrix_A[24] = {1,0,0,0, 1,1,1,1, 1,-1,1,-1, 1,2,4,8, 1,-2,4,-8, 0,0,0,1};
       A = (float*)_mm_malloc(sizeof(float)*24, 32);
       memcpy(A, matrix_A, 24*sizeof(float));
    }

    if(m == 6 && r == 3)
    {
       float matrix_Bt[64] = {1,0,-5.25,0,5.25,0,-1,0, 
                              0,1,1,-4.25,-4.25,1,1,0,
                              0,-1,1,4.25,-4.25,-1,1,0,
                              0,0.5,0.25,-2.5,-1.25,2,1,0,
                              0,-0.5,0.25,2.5,-1.25,-2,1,0,
                              0,2,4,-2.5,-5,0.25,1,0,
                              0,-2,4,2.5,-5,-0.5,1,0,
                              0,-1,0,5.25,0,-5.25,0,1
                              };
       Bt = (float*)_mm_malloc(sizeof(float)*64, 32);
       memcpy(Bt, matrix_Bt, 64*sizeof(float));

       float matrix_B[64]  = {1,0,0,0,0,0,0,0,
                              0,1,-1,0.5,-0.5,2,-2,-1,
                              -5.25,1,1,0.25,0.25,4,4,0,
                              0,-4.25,4.25,-2.5,2.5,-2.5,2.5,5.25,
                              5.25,-4.25,-4.25,-1.25,-1.25,-5,-5,0,
                              0,1,-1,2,-2,0.5,-0.5,-5.25,
                              -1,1,1,1,1,1,1,0,
                              0,0,0,0,0,0,0,1
                              };
       B = (float*)_mm_malloc(sizeof(float)*64, 32);
       memcpy(B, matrix_B, 64*sizeof(float));

       float  matrix_G[24] = {1,0,0,
                              -0.2222222222,-0.2222222222,-0.2222222222,
                              -0.2222222222,0.2222222222,-0.2222222222,
                              0.01111111111,0.02222222222, 0.04444444444,
                              0.01111111111,-0.02222222222, 0.04444444444,
                              0.71111111111,0.35555555555, 0.17777777777,
                              0.71111111111,-0.35555555555, 0.17777777777,
                              0,0,1
                             };
       G = (float*)_mm_malloc(sizeof(float)*24, 32);
       memcpy(G, matrix_G, 24*sizeof(float));

       float matrix_Gt[24] = {1,-0.2222222222,-0.2222222222,0.01111111111,0.01111111111,0.71111111111,0.71111111111,0,
                               0,-0.2222222222,0.2222222222,0.02222222222,-0.02222222222,0.35555555555,-0.35555555555,0,
                               0,-0.2222222222,-0.2222222222,0.04444444444,0.04444444444,0.17777777777,0.17777777777,1
                             };
       Gt = (float*)_mm_malloc(sizeof(float)*24, 32);
       memcpy(Gt, matrix_Gt, 24*sizeof(float));


       float matrix_At[48] = {1,1,1,1,1,1,1,0,
                              0,1,-1,2,-2,0.5,-0.5,0,
                              0,1,1,4,4,0.25,0.25,0,
                              0,1,-1,8,-8,0.125,-0.125,0,
                              0,1,1,16,16,0.0625,0.0625,0,
                              0,1,-1,32,-32,0.03125,-0.03125,1
                             };
       At = (float*)_mm_malloc(sizeof(float)*48, 32);
       memcpy(At, matrix_At, 48*sizeof(float));

       float  matrix_A[48] = {1,0,0,0,0,0,
                              1,1,1,1,1,1,
                              1,-1,1,-1,1,-1,
                              1,2,4,8,16,32,
                              1,-2,4,-8,16,-32,
                              1,0.5,0.25,0.125,0.0625,0.03125,
                              1,-0.5,0.25,-0.125,0.0625,-0.03125,
                              0,0,0,0,0,1
                             };
       A = (float*)_mm_malloc(sizeof(float)*48, 32);
       memcpy(A, matrix_A, 48*sizeof(float));
    }
}

void Wino::createU(Mat& kernel_blob)
{
    int ch = kernel_blob.c;

    out_ch = ch / in_ch;
    
    U = (float*) _mm_malloc(sizeof(float) * t * t * ch, 32);

//    float* Gg = (float*) _mm_malloc(sizeof(float) * t * r, 32);

    float sum = 0;

#pragma omp parallel for
    for(int cc = 0; cc < ch ; cc++)
    {
        float Gg[t*r];
        float* g = (float*)kernel_blob + cc * r * r;
        float* Ux = U + cc * t * t;
        for (int i = 0; i < t; i++)
            for (int j=0; j< r; j++)
            {
                sum = 0.0;
                for(int k=0; k < r; k++)
                {
                    sum += G[i*r + k] * g[k*r+j];
                }
                Gg[i*r+j] = sum;
            }


        for(int i=0; i< t; i++)
            for (int j=0; j< t; j++)
            {
                sum = 0.0;
                for(int k = 0; k<r; k++)
                    sum += Gg[i* r + k] * Gt[k*t+j];
                Ux[i*t+j] = sum;
            }

/*      
        if(cc==0)
        {
//            printMatrix("Gg in hp", Gg, 4,3);
//            printMatrix("Kernel in hp", g, 3,3);
//            printMatrix("Gt in hp", Gt, 3,4);
//            printMatrix("U in hp", Ux, 4,4);
        }
*/        
    } 
//    _mm_free(Gg);

//    printMatrix("U in hp", U, 4,4);
}


void Wino::convertU_A()
{
//    U = (float*) _mm_malloc(sizeof(float) * t * t * ch, 32);    

    U_A = (float*) _mm_malloc(sizeof(float) * t * t * out_ch * in_ch, 32);

#pragma omp parallel for
    for(int ik=0; ik < t*t; ik++)
    {
        float* tU_A = U_A + in_ch * out_ch * ik;
        for(int occ=0; occ < out_ch; occ++)
        {
            float* xU_A = tU_A + in_ch * occ;
            for(int icc=0; icc< in_ch; icc++)
                xU_A [icc] = U[t*t*in_ch*occ + t*t*icc + ik];

        }
    }

//    printMatrix("U", U, 1,16);
//    printMatrix("U_A", U_A, 1,16);

}

void Wino::convertU_B()
{
	U_B = U;
}


void Wino::convertU_C()
{
    U_C = (float*) _mm_malloc(sizeof(float) * t * t * out_ch * in_ch, 32);

#pragma omp parallel for
    for (int ik = 0; ik < t * t; ik++)
        for(int occ = 0; occ < out_ch / 8; occ++)
            for (int icc = 0; icc < in_ch; icc++)
                for(int it = 0; it < 8; it ++)
                    U_C[in_ch * out_ch * ik + 8 * in_ch * occ + 8*icc + it] = U[in_ch * t * t * (occ * 8+it) + t * t * icc + ik];


/*
    printMatrix("U C ", U_C, 1,1);
    printMatrix("U C ", U_C + in_ch * out_ch, 1,1);
    printMatrix("U C ", U_C + in_ch * out_ch * 2, 1,1);
    printMatrix("U C ", U_C + in_ch * out_ch * 3, 1,1);
*/


}




void Wino::createV(Mat& bottom_blob)
{
    in_w = bottom_blob.w;
    in_h = bottom_blob.h;
    in_ch = bottom_blob.c;

    printf(" in_w = %d, in_h = %d, in_ch = %d \n", in_w, in_h, in_ch);

//    numTiles_w = (in_w - (r - 1)) / m;
//    numTiles_h = (in_h - (r - 1)) / m;

    int tmp_w = in_w - (r - 1);
    int tmp_h = in_h - (r - 1);

    numTiles_w = (tmp_w % m == 0) ? tmp_w/m : (tmp_w/m+1);
    numTiles_h = (tmp_h % m == 0) ? tmp_h/m : (tmp_h/m+1);

    numTiles = numTiles_w * numTiles_h;

//    out_w = numTiles_w * m;
//    out_h = numTiles_h * m;

    V = (float*) _mm_malloc(sizeof(float) * numTiles_w * numTiles_h * t * t * in_ch, 32);
    float* Btd_all = (float*) _mm_malloc(sizeof(float) * t * t * 8, 32);

//    float sum = 0.0;



    if(tmp_w % m == 0)
    { 
#pragma omp parallel for
        for (int cc = 0; cc < in_ch; cc ++)
        {
            float sum=0.0;
            float* Vx = V + cc * numTiles_w * numTiles_h * t * t;
            float* d = (float*) bottom_blob.channel(cc);
    //        int thread_idx=omp_get_thread_num();
    //        float* Btd = Btd_all + thread_idx * t * t;
            float Btd[t*t];
            for(int i =0; i < numTiles_h; i++)
            {
                for (int j=0; j< numTiles_w; j++)
                {
                    for(int kk=0; kk < t; kk++)
                        for(int ss=0; ss < t; ss++)
                        {
                            sum = 0.0;
                            for(int pp=0; pp< t; pp++)
                            {
                                sum += Bt[kk*t + pp] * d[(i*m+pp)* in_w + j*m + ss];

    //                            if(cc==0 && i==0 && j==0 && kk==0 && ss==0)
    //                                std::cout<<".. "<< Bt[kk*t+pp] <<" "<< d[(i*r+pp)*in_w + j*r+ss]<<endl;
                            }
    /*                      
                            if(cc==0 && i==0 && j==0 && kk==0 && ss==0)
                                std::cout<<" sum = "<< sum<<endl;
    */                            
                            Btd[kk*t+ss] = sum;
                        }
    /*                
                    if(cc==2 && i==0 && j==0)
                        printMatrix("Btd in hp", Btd, 4,4);
    */

                    for(int kk=0; kk< t; kk++)
                        for(int ss=0; ss<t; ss++)
                        {
                            sum = 0.0;
                            for(int pp=0; pp<t; pp++)
                                sum+= Btd[kk*t+pp] * B[pp*t+ss];
                            Vx[(i*numTiles_w+j)*t*t + kk*t+ss] = sum;
                        }
    /*
                    if(cc==2 && i==0 && (j==0 || j==1))
                    {
            //            printMatrix("B in hp", B, 4,4);
                        printMatrix("Vx in hp", Btd, 4,4);
                    }
    */                
                }

            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int cc = 0; cc < in_ch; cc ++)
        {
            float sum=0.0;
            float* Vx = V + cc * numTiles_w * numTiles_h * t * t;
            float* d = (float*) bottom_blob.channel(cc);
            float Btd[t*t];
            for(int i =0; i < numTiles_h; i++)
            {
                for (int j=0; j< numTiles_w; j++)
                {
                    for(int kk=0; kk < t; kk++)
                        for(int ss=0; ss < t; ss++)
                        {
                            sum = 0.0;
                            for(int pp=0; pp< t; pp++)
                            {
                                int colPos = i*m + pp;
                                int rowPos = j*m + ss;

                                float dx = 0;

                                if(colPos >= in_h || rowPos >= in_w )
                                {
                                    dx = 0;
//                                    printf(" if");
//    printf(" in_w = %d, in_h = %d, in_ch = %d \n", in_w, in_h, in_ch);
//                                printf(" dx = %f, colPos = %d, rowPos = %d, in_w = %d, in_h = %d, cc= %d, i = %d, j = %d, kk= %d, ss = %d, pp = %d\n", dx, colPos, rowPos, in_w, in_h, cc, i, j, kk, ss, pp);
                                }
                                else
                                {
                                    dx = d[colPos * in_w + rowPos];
//                                    printf(" else");
                                }
//                                printf(" dx = %f, colPos = %d, rowPos = %d, in_w = %d, in_h = %d\n", dx, colPos, rowPos, in_w, in_h);

//                                sum += Bt[kk*t + pp] * d[(i*m+pp)* in_w + j*m + ss];
                                sum += Bt[kk*t + pp] * dx;

                            }
                            Btd[kk*t+ss] = sum;
                        }

                    for(int kk=0; kk< t; kk++)
                        for(int ss=0; ss<t; ss++)
                        {
                            sum = 0.0;
                            for(int pp=0; pp<t; pp++)
                                sum+= Btd[kk*t+pp] * B[pp*t+ss];
                            Vx[(i*numTiles_w+j)*t*t + kk*t+ss] = sum;
                        }
                }

            }
        }

    }
/*
    printMatrix("V in hp", V, 4,4);

    printMatrix("V in hp", V, 1,1);
    printMatrix("V in hp", V +  t*t*numTiles, 1,1);
    printMatrix("V in hp", V + t*t*numTiles*2, 1,1);
    printMatrix("V in hp", V + t*t*numTiles*3, 1,1);
//    printMatrix("Vx-seg in hp", V + t * t * numTiles, 4,8);

*/

    _mm_free(Btd_all);
}

void Wino::convertV_A()
{
    V_A = (float*) _mm_malloc(sizeof(float) * numTiles_w * numTiles_h * t * t * in_ch, 32);

#pragma omp parallel for
    for(int it = 0; it < numTiles_w * numTiles_h; it ++)
    {
        float* tV_A = V_A + t * t * in_ch * it;

        for(int ih = 0; ih < t; ih ++)
            for(int iw =0; iw < t; iw ++)
            {
                float* xV_A = tV_A + (ih*t+iw) * in_ch;
                for(int icc=0; icc < in_ch; icc++)
                    xV_A[icc] = V[t*t*numTiles*icc + t*t*it + ih*t+iw];
            }
    }

//    printMatrix("V_A", V_A, 1,16);


}


void Wino::convertV_B()
{
    V_B = (float*) _mm_malloc(sizeof(float) * numTiles_w * numTiles_h * t * t * in_ch, 32);

#pragma omp parallel for
	for(int it = 0; it < numTiles_w * numTiles_h; it ++)
	{
		float* tV_B = V_B + t * t * in_ch * it;
		for (int icc = 0; icc < in_ch; icc++)
		{
			float* xV_B = tV_B + t * t * icc;
			for(int ihw = 0; ihw < t * t; ihw ++)
			{
				xV_B[ihw] = V[t*t*numTiles*icc + t * t * it + ihw];
			}
		}
	}

//    printMatrix("V_B in hp_B", V_B, 4,8);

}


void Wino::convertV_C()
{
    V_C = (float*) _mm_malloc(sizeof(float) * numTiles_w * numTiles_h * t * t * in_ch, 32);

#pragma omp parallel for
    for(int ik = 0; ik < t*t; ik ++)
        for(int it = 0; it < numTiles; it ++)
            for(int icc = 0; icc < in_ch; icc ++)
                V_C[in_ch * numTiles * ik + in_ch * it + icc] = V[t * t * numTiles * icc + t * t * it + ik];

//    printMatrix("V C ", V_C, 1,4);
//    printMatrix("V C ", V_C + in_ch * numTiles, 1,1);
//    printMatrix("V C ", V_C + in_ch * numTiles * 2, 1,1);
//    printMatrix("V C ", V_C + in_ch * numTiles * 3, 1,1);

}




void Wino::getNaiveResult(Mat& top_blob)
{
//    top_blob.create(out_w, out_h, out_ch);

    int ostep = alignSize(out_w * out_h * sizeof(float), 16) >> 2;

//    float* result=(float*)_mm_malloc(sizeof(float) * numTiles* m * m * out_ch, 32);
    float* result=(float*)_mm_malloc(sizeof(float) * (ostep * out_ch + m), 32);

    memset(result, 0, sizeof(float)* (ostep * out_ch + m));
//    memset(top_blob.data, 0, sizeof(float) * out_w * out_h * out_ch );

//    printf(" out_w = %d, out_h = %d, out_ch = %d\n", out_w, out_h, out_ch);


#pragma omp parallel for
    for (int occ=0; occ < out_ch; occ++)
    {
        float tmp[t*t];
        float tmp2[m*t];
//        float* refOut = (float*)top_blob.channel(occ);
        float* refOut = result + ostep * occ;
//        float* refOut = result + numTiles * m * m * occ;
        float* refU = U + t * t * in_ch * occ;
        for(int icc=0; icc<in_ch; icc++)
        {
            float* tV = V + numTiles * t * t * icc;
            float* tU = refU + t * t * icc;

            for(int ohh=0; ohh < numTiles_h; ohh++)
                for(int oww=0; oww < numTiles_w; oww++)
                {
//                    float* tOut = refOut + (ohh*numTiles_w + oww) * m * m;
                    float* tOut = refOut + ohh * m * out_w + oww * m;
                    float* ttV = tV + (ohh * numTiles_w + oww) * t * t;

//                    for(int itmp = 0; itmp < t*t; itmp ++)
//                        tmp[itmp] = tU[itmp] * ttV[itmp];

                    for(int itmp=0; itmp < m; itmp++)
                        for(int jtmp=0; jtmp < t; jtmp ++)
                        {
                            float sum=0.0;
                            for(int ktmp=0; ktmp< t; ktmp++)
//                                sum+= At[itmp * t + ktmp] * tmp[ktmp* t + jtmp];
                                sum+= At[itmp * t + ktmp] * tU[ktmp* t + jtmp] * ttV[ktmp*t+jtmp];
                            tmp2[itmp * t + jtmp] = sum;
                        }
                    
                    for(int itmp=0; itmp < m; itmp++)
                        for(int jtmp=0; jtmp< m; jtmp++)
                        {
                            float sum=0.0;
                            for(int ktmp=0; ktmp<t; ktmp++)
                                sum += tmp2[itmp * t + ktmp] * A[ktmp * m + jtmp];
//                            tOut[itmp * m + jtmp] += sum;
                            if((oww*m+jtmp) < out_w && (ohh*m +itmp) < out_h)
                                tOut[itmp * out_w + jtmp] += sum;
                        }
                    if(occ==0 && icc==1 && ohh==0 && (oww==0 || oww == 1))
                    {
//                        printMatrix("U", tU, 4,4);
//                        printf(" oww=%d\n", oww);
//                        printMatrix("V", ttV, 4,4);
//                        printMatrix("AtUV", tmp2, 2,4);
//                        printMatrix("UV", tmp, 4,4);
//                        printMatrix("re", tOut, 1,2);
//                        printf(" Distance = %d\n", (refOut-top_blob.data));

                    }
                }
        }
    }

    top_blob = Mat(out_w, out_h, out_ch, result);
}




void Wino::getResult_A(Mat& top_blob)
{
    UV_A = (float*) _mm_malloc(sizeof(float) * numTiles * t *t * out_ch, 32);
    tileR = (float*) _mm_malloc(sizeof(float) * numTiles * m * m * out_ch, 32);

    double t1, t2, t3, t4;
    t1 = microtime();


//#pragma omp parallel for
//#pragma ivdep
//    for(int it=0; it < numTiles; it++)




#pragma omp parallel for
    for(int it=0; it < numTiles; it++)
    {
/*
    zmmd_t regU;
    zmmd_t regV;
    zmmd_t regSum;
*/
        for(int ik=0; ik < t*t; ik++)
        {
//            float tmp[t*t];
            for(int occ=0; occ< out_ch; occ++)
            {
                float sum=0.0;
//                regSum.regps = _mm256_set1_ps(0);
//#pragma vector always
//                for(int icc=0; icc < in_ch; icc++)
                for(int icc=0; icc < in_ch; icc+=1)
                {
                    sum += U_A[in_ch*out_ch*ik + in_ch * occ + icc] * V_A[t*t*in_ch*it + in_ch*ik + icc];
/*
                    regU.regps = _mm256_load_ps(U_A + in_ch * out_ch*ik + in_ch * occ + icc);
                    regV.regps = _mm256_load_ps(V_A + t * t * in_ch * it + in_ch * ik + icc);

                    regSum.regps = _mm256_fmadd_ps(regU.regps, regV.regps, regSum.regps);
*/
                }
	//			printMatrix("regSum", regSum.elemsps, 4, 2);

//				sum = regSum.elemsps[0] + regSum.elemsps[1] + regSum.elemsps[2] + regSum.elemsps[3] + regSum.elemsps[4] + regSum.elemsps[5] + regSum.elemsps[6] + regSum.elemsps[7];
	//			printf(" sum = %f \n", sum);
                
                UV_A[ out_ch * t * t * it + out_ch * ik + occ] = sum;
            }
        }

        for(int occ=0; occ < out_ch; occ++)
        {
            float tmp[m*t];
            for( int ih=0; ih < m ; ih++)
                for( int iw=0; iw < t ; iw++)
                {
                    float sum=0.0;
                    for(int ik =0; ik < t; ik++)
                        sum += At[ih*t + ik ] * UV_A[out_ch * t *t*it+out_ch * (ik*t+iw) + occ];
                    tmp[ih*t+iw] = sum;
                }
           for(int ih=0; ih<m; ih++)
              for(int iw=0; iw < m; iw++)
              {
                  float sum=0.0;
                  for(int ik=0; ik < t; ik++)
                      sum+= tmp[ih*t+ik] * A[ik*m+iw];
                  tileR[out_ch * m * m * it + out_ch * (ih*m + iw) + occ] = sum;
              } 
        }
    }
//    printMatrix("tileR", tileR, 1,16);
    t2 = microtime();
    std::cout<<" part1 : "<< t2 - t1<<endl;

/*	
    printMatrix("tileR in hp_A", tileR, 1, 1);
    printMatrix("tileR in hp_A", tileR+in_ch, 1, 1);
    printMatrix("tileR in hp_A", tileR+in_ch * 2, 1, 1);
    printMatrix("tileR in hp_A", tileR+in_ch * 3, 1, 1);
    printMatrix("tileR in hp_A", tileR+in_ch * 4, 1, 1);
    printMatrix("tileR in hp_A", tileR+in_ch * 5, 1, 1);
    printMatrix("tileR in hp_A", tileR+in_ch * 6, 1, 1);
    printMatrix("tileR in hp_A", tileR+in_ch * 7, 1, 1);
	printMatrix("UV_A", UV_A, 4, 4);
*/
    
    t3 = microtime();

//    top_blob.create(out_w, out_h, out_ch);
    int ostep = alignSize(out_w * out_h * sizeof(float), 16) >> 2;
    float* result=(float*)_mm_malloc(sizeof(float) * (ostep * out_ch + m), 32);
    memset(result, 0, sizeof(float)* (ostep * out_ch + m));
//    memset(top_blob.data, 0, sizeof(float) * out_w * out_h * out_ch);

#pragma omp parallel for
    for(int occ=0; occ < out_ch; occ++)
    {
//        float* refOut = (float*)top_blob.channel(occ);
        float* refOut = result + ostep * occ;
        for(int ih = 0; ih < numTiles_h; ih ++)
            for(int iw =0 ;iw < numTiles_w; iw ++)
            {
                float* vOut = refOut + ih * m * out_w + iw * m;
                for(int is=0; is < m; is++)
                    for(int it=0; it<m; it++)
                        if((iw*m+it) < out_w && (ih*m +is) < out_h)
                        vOut[is*out_w + it] = tileR[out_ch * m * m *(ih*numTiles_w + iw) + out_ch * (is * m + it) + occ];

            }
    }

    top_blob = Mat(out_w, out_h, out_ch, result);

    t4 = microtime();
    std::cout<<" part2 : "<< t4 - t3<<endl;

//    printMatrix("refOut", top_blob.data, 1,16);
}


void Wino::getResult_B(Mat& top_blob)
{

    double tx, ty;
    tx = microtime();
    UV_B = (float*) _mm_malloc(sizeof(float) * numTiles * t *t * out_ch, 64);
    tileR = (float*) _mm_malloc(sizeof(float) * numTiles * m * m * out_ch, 64);

    int ostep = alignSize(out_w * out_h * sizeof(float), 16) >> 2;
    float* result=(float*)_mm_malloc(sizeof(float) * (ostep * out_ch + m), 64);
    memset(result, 0, sizeof(float)* (ostep * out_ch + m ));
//    memset(top_blob.data, 0, sizeof(float) * out_w * out_h * out_ch);

    ty = microtime();
    std::cout<<" part0 is "<< ty - tx << endl;
	

    double t1, t2, t3, t4;
    t1 = microtime();

#pragma omp parallel for
	for (int occ = 0; occ < out_ch; occ ++)
	{
		for(int it = 0; it < numTiles; it ++)
		{
/*            
            zmmd_t regU;
            zmmd_t regV;
            zmmd_t regUV;
            
            zmmd_t regU2;
            zmmd_t regV2;
            zmmd_t regUV2;
*/            
//            zmmd_t regSum;
//            regSum.regps = _mm256_set1_ps(0);
			for (int icc = 0; icc < in_ch; icc ++)
				for (int ik = 0; ik < t*t; ik ++)
				{

/*
                    regU.regps = _mm256_load_ps(U_B + t * t * in_ch * occ + t * t * icc);
                    regV.regps = _mm256_load_ps(V_B + t * t * in_ch * it + t * t * icc);
                    regUV.regps= _mm256_load_ps(UV_B + t * t * numTiles * occ + t * t * it);

                    regUV.regps = _mm256_fmadd_ps(regU.regps, regV.regps, regUV.regps);
                    
                    regU2.regps = _mm256_load_ps(U_B + t * t * in_ch * occ + t * t * icc + 8);
                    regV2.regps = _mm256_load_ps(V_B + t * t * in_ch * it + t * t * icc + 8);
                    regUV2.regps= _mm256_load_ps(UV_B + t * t * numTiles * occ + t * t * it + 8);

                    regUV2.regps = _mm256_fmadd_ps(regU.regps, regV.regps, regUV.regps);
                    _mm256_store_ps(UV_B + t * t * numTiles * occ + t * t * it, regUV.regps);
                    _mm256_store_ps(UV_B + t * t * numTiles * occ + t * t * it + 8, regUV2.regps);
*/

					UV_B[t * t * numTiles * occ + t * t * it + ik] += U_B[t * t * in_ch * occ + t * t * icc + ik] * V_B[t * t * in_ch * it + t * t * icc + ik];
				}
		}
        
		for(int it = 0; it < numTiles; it ++)
		{
            float tmp[m*t];
            for( int ih=0; ih < m ; ih++)
                for( int iw=0; iw < t ; iw++)
                {
                    float sum=0.0;
                    for(int ik =0; ik < t; ik++)
                        sum += At[ih*t + ik ] * UV_B[t * t * numTiles * occ + t * t * it + (ik*t+iw)];
                    tmp[ih*t+iw] = sum;
                }
            for(int ih=0; ih<m; ih++)
                for(int iw=0; iw < m; iw++)
                {
                    float sum=0.0;
                    for(int ik=0; ik < t; ik++)
                       sum+= tmp[ih*t+ik] * A[ik*m+iw];
                    tileR[m * m * numTiles * occ + m * m * it + (ih*m + iw)] = sum;
                } 
		}
	}

    t2 = microtime();
    std::cout<<" part1 : "<< t2 - t1<<endl;
/* 
	printMatrix("tileR in hp_B", tileR, 2, 2);
	printMatrix("tileR in hp_B", tileR+4, 2, 2);
	printMatrix("UV_B", UV_B, 4, 4);
*/  
    t3 = microtime();


//#pragma omp parallel for
    for(int occ=0; occ < out_ch; occ++)
    {
//        float* refOut = (float*)top_blob.channel(occ);
        float* refOut = result + ostep * occ;
        for(int ih = 0; ih < numTiles_h; ih ++)
            for(int iw =0 ;iw < numTiles_w; iw ++)
            {
                float* vOut = refOut + ih * m * out_w + iw * m;
                for(int is=0; is < m; is++)
                    for(int it=0; it<m; it++)
                        if((iw*m+it) < out_w && (ih*m +is) < out_h)
						{
                        	vOut[is*out_w + it] = tileR[m * m * numTiles * occ + m * m *(ih*numTiles_w + iw)+ (is * m + it)];
//							printf(" vOut[%d] = %lf", is*out_w + it, vOut[is*out_w + it]);
						}
//						else
//							printf("occ = %d, it = %d, ih = %d, iw = %d, is = %d \n", occ, it, ih, iw, is);
//				printMatrix("Result:", vOut, 2,2);

            }
    }

    top_blob = Mat(out_w, out_h, out_ch, result);

    t4 = microtime();
    std::cout<<" part2 : "<< t4 - t3<<endl;


}


void Wino::getResult_C(Mat& top_blob)
{

    double tx, ty;
    tx = microtime();
    UV_C = (float*) _mm_malloc(sizeof(float) * numTiles * t *t * out_ch, 64);
//    memset(UV_C, 0, sizeof(float)* (numTiles * t * t * out_ch));
    tileR = (float*) _mm_malloc(sizeof(float) * numTiles * m * m * out_ch, 64);

    int ostep = alignSize(out_w * out_h * sizeof(float), 16) >> 2;
    float* result=(float*)_mm_malloc(sizeof(float) * (ostep * out_ch + m), 64);
    memset(result, 0, sizeof(float)* (ostep * out_ch + m ));
//    memset(top_blob.data, 0, sizeof(float) * out_w * out_h * out_ch);

    ty = microtime();
    std::cout<<" part0 is "<< ty - tx << endl;
	

    double t1, t2, t3, t4;
    t1 = microtime();

#pragma omp parallel for
    for(int ik = 0; ik < t * t; ik ++)
        for(int it = 0; it < numTiles; it ++)
            for(int occ=0; occ < out_ch/8; occ ++)
            {
                zmmd_t regU;
                zmmd_t regV;
                zmmd_t regUV;
//                regUV.regps= _mm256_load_ps(UV_C + out_ch * numTiles * ik + out_ch * it + 8 * occ);
                regUV.regps= _mm256_set1_ps(0);
                for(int icc=0; icc<in_ch; icc++)
                {
                    regU.regps = _mm256_load_ps(U_C + in_ch * out_ch * ik + 8 * in_ch * occ + 8 * icc);
                    regV.regps = _mm256_set1_ps(*(V_C + in_ch * numTiles * ik + in_ch * it + icc));


                    regUV.regps = _mm256_fmadd_ps(regU.regps, regV.regps, regUV.regps);

//                    for(int ig=0; ig < 8; ig++)
//                        UV_C[out_ch * numTiles * ik + out_ch * it + 8 * occ + ig] += U_C[in_ch * out_ch * ik + 8 * in_ch * occ + 8 * icc + ig] * V_C[in_ch * numTiles * ik + in_ch * it + icc];
                }
                _mm256_store_ps(UV_C + out_ch * numTiles * ik + out_ch * it + 8 * occ, regUV.regps);
            }

/*
	printMatrix("UV_C", UV_C, 1, 1);
	printMatrix("UV_C", UV_C + out_ch * numTiles, 1, 1);
	printMatrix("UV_C", UV_C + out_ch * numTiles * 2, 1, 1);
	printMatrix("UV_C", UV_C + out_ch * numTiles * 3, 1, 1);
*/

    t2 = microtime();
    std::cout<<" part1-1: "<< t2 - t1<<endl;

    t1 = microtime();

#pragma omp parallel for
	for (int occ = 0; occ < out_ch; occ ++)
	{
		for(int it = 0; it < numTiles; it ++)
		{
            float tmp[m*t];
            for( int ih=0; ih < m ; ih++)
                for( int iw=0; iw < t ; iw++)
                {
                    float sum=0.0;
                    for(int ik =0; ik < t; ik++)
                        sum += At[ih*t + ik ] * UV_C[out_ch * numTiles * (ik*t+iw) + out_ch * it + occ];
                    tmp[ih*t+iw] = sum;
                }
            for(int ih=0; ih<m; ih++)
                for(int iw=0; iw < m; iw++)
                {
                    float sum=0.0;
                    for(int ik=0; ik < t; ik++)
                       sum+= tmp[ih*t+ik] * A[ik*m+iw];
                    tileR[m * m * numTiles * occ + m * m * it + (ih*m + iw)] = sum;
                } 
		}
	}

/*    
	printMatrix("tileR in hp_C", tileR, 2, 2);
	printMatrix("tileR in hp_C", tileR+4, 2, 2);
	printMatrix("UV_C", UV_C, 4, 4);
*/


    t2 = microtime();
    std::cout<<" part1-2 : "<< t2 - t1<<endl;


    t3 = microtime();


//#pragma omp parallel for
    for(int occ=0; occ < out_ch; occ++)
    {
//        float* refOut = (float*)top_blob.channel(occ);
        float* refOut = result + ostep * occ;
        for(int ih = 0; ih < numTiles_h; ih ++)
            for(int iw =0 ;iw < numTiles_w; iw ++)
            {
                float* vOut = refOut + ih * m * out_w + iw * m;
                for(int is=0; is < m; is++)
                    for(int it=0; it<m; it++)
                        if((iw*m+it) < out_w && (ih*m +is) < out_h)
						{
                        	vOut[is*out_w + it] = tileR[m * m * numTiles * occ + m * m *(ih*numTiles_w + iw)+ (is * m + it)];
//							printf(" vOut[%d] = %lf", is*out_w + it, vOut[is*out_w + it]);
						}
//						else
//							printf("occ = %d, it = %d, ih = %d, iw = %d, is = %d \n", occ, it, ih, iw, is);
//				printMatrix("Result:", vOut, 2,2);

            }
    }

    top_blob = Mat(out_w, out_h, out_ch, result);

    t4 = microtime();
    std::cout<<" part2 : "<< t4 - t3<<endl;


}










int benchmark_hp_winograd_Naive(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output)
{

    int m = 2;
    int r = 3;

    double t1,t2,t3,t4;

    Wino* wino = new Wino(m, r);

    wino->out_w=(bottom_blob.w+ 2 * pad_w - (dilation_w * (kernel_size - 1) + 1)) / stride_w + 1;
    wino->out_h=(bottom_blob.h+ 2 * pad_h - (dilation_h * (kernel_size - 1) + 1)) / stride_h + 1;

    t1 = microtime();
    wino->createV(bottom_blob);
    t2 = microtime();

    std::cout<<" Create matrix V Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->createU(kernel_blob);
    t2 = microtime();

    std::cout<<" Create matrix U Time: "<< t2 - t1 <<std::endl;
    
    t1 = microtime();
    wino->getNaiveResult(top_blob);
    t2 = microtime();

    std::cout<<" get Result Time: "<< t2 - t1 <<std::endl;

    return 0;

}



int benchmark_hp_winograd_A(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output)
{

    int m = 2;
    int r = 3;

    double t1,t2,t3,t4;

    Wino* wino = new Wino(m, r);

    wino->out_w=(bottom_blob.w+ 2 * pad_w - (dilation_w * (kernel_size - 1) + 1)) / stride_w + 1;
    wino->out_h=(bottom_blob.h+ 2 * pad_h - (dilation_h * (kernel_size - 1) + 1)) / stride_h + 1;

    t1 = microtime();
    wino->createV(bottom_blob);
    t2 = microtime();

    std::cout<<" Create matrix V Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->createU(kernel_blob);
    t2 = microtime();

    std::cout<<" Create matrix U Time: "<< t2 - t1 <<std::endl;
    
    t1 = microtime();
    wino->convertU_A();
    t2 = microtime();

    std::cout<<" convertU_A Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->convertV_A();
    t2 = microtime();

    std::cout<<" convertV_A Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->getResult_A(top_blob);
    t2 = microtime();

    std::cout<<" getResult_A Time: "<< t2 - t1 <<std::endl;

    return 0;
}


int benchmark_hp_winograd_B(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output)
{

    int m = 2;
    int r = 3;

    double t1,t2,t3,t4;

    Wino* wino = new Wino(m, r);

    wino->out_w=(bottom_blob.w+ 2 * pad_w - (dilation_w * (kernel_size - 1) + 1)) / stride_w + 1;
    wino->out_h=(bottom_blob.h+ 2 * pad_h - (dilation_h * (kernel_size - 1) + 1)) / stride_h + 1;

    t1 = microtime();
    wino->createV(bottom_blob);
    t2 = microtime();

    std::cout<<" Create matrix V Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->createU(kernel_blob);
    t2 = microtime();

    std::cout<<" Create matrix U Time: "<< t2 - t1 <<std::endl;
    
    t1 = microtime();
    wino->convertU_B();
    t2 = microtime();

    std::cout<<" convertU_B Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->convertV_B();
    t2 = microtime();

    std::cout<<" convertV_B Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->getResult_B(top_blob);
    t2 = microtime();

    std::cout<<" getResult_B Time: "<< t2 - t1 <<std::endl;

    return 0;
}


int benchmark_hp_winograd_C(Mat& bottom_blob, Mat& kernel_blob, Mat& top_blob, Mat& bias_data, int pad_w, int pad_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int kernel_size, int num_output)
{

    int m = 2;
    int r = 3;

    double t1,t2,t3,t4;

    Wino* wino = new Wino(m, r);

    wino->out_w=(bottom_blob.w+ 2 * pad_w - (dilation_w * (kernel_size - 1) + 1)) / stride_w + 1;
    wino->out_h=(bottom_blob.h+ 2 * pad_h - (dilation_h * (kernel_size - 1) + 1)) / stride_h + 1;

    t1 = microtime();
    wino->createV(bottom_blob);
    t2 = microtime();

    std::cout<<" Create matrix V Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->createU(kernel_blob);
    t2 = microtime();

    std::cout<<" Create matrix U Time: "<< t2 - t1 <<std::endl;
    
    t1 = microtime();
    wino->convertU_C();
    t2 = microtime();

    std::cout<<" convertU_C Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->convertV_C();
    t2 = microtime();

    std::cout<<" convertV_C Time: "<< t2 - t1 <<std::endl;

    t1 = microtime();
    wino->getResult_C(top_blob);
    t2 = microtime();

    std::cout<<" getResult_C Time: "<< t2 - t1 <<std::endl;

    return 0;
}





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

    int inw = 228;
    int inh = 228;
    int inch = 64;

    int outch = 64;

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
//	checkResults(top_gemm_blob.data, xtop_blob);

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

