#include <stdio.h>
#include <stdlib.h>


void printMatrix(const char* str, float* a, int m, int n)
{
   int i,j;

   printf(" Matrix %s \n", str);
   for(i = 0; i< m; i++)
   {
      for(j=0; j<n; j++)
        printf(" %f ", a[i*n+j]);

      printf("\n");
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

float* matrixMul(float* a, float* b, int m, int k, int n)
{
   float* re = (float*) malloc( sizeof(float) * m * n);

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

float* matrixDotProduct(float* a, float*b, int m, int n)
{
   float* re = (float*) malloc(sizeof(float) * m * n);
 
   int i,j;

   for(i = 0; i<m; i++)
      for(j=0; j<n; j++)
        re[i*n+j] = a[i*n+j] * b[i*n+j];

   return re;
}

float* winograd_m2r3(float* matrix_d, float* matrix_g)
{
 
   float matrix_Bt[16] = {1,0,-1,0, 0,1,1,0, 0,-1,1,0, 0,1,0,-1};
   float* matrix_B = matrixTranspose(matrix_Bt, 4,4);

   float matrix_G[12] = {1,0,0, 0.5,0.5,0.5, 0.5,-0.5,0.5, 0,0,1};
   float* matrix_Gt = matrixTranspose(matrix_G, 4,3);

   float matrix_At[8] = {1,1,1,0, 0,1,-1,-1};
   float* matrix_A = matrixTranspose(matrix_At, 2,4);

   float* matrix_U = matrixMul(matrixMul(matrix_G, matrix_g, 4,3,3), matrix_Gt, 4,3,4);
   float* matrix_V = matrixMul(matrixMul(matrix_Bt, matrix_d, 4,4,4), matrix_B, 4,4,4);
   float* matrix_UV = matrixDotProduct(matrix_U, matrix_V, 4,4);

   float* re = matrixMul(matrixMul(matrix_At, matrix_UV, 2,4,4), matrix_A, 2,4,2);

   return re;  
}

float* conv_direct(float* d, float* g)
{

   int d_w = 4;
   int d_h = 4;
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
      }

    return re;

}

int main(int argc, char** argv)
{
   float matrix_d[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
   float matrix_g[9]  = {1,1,1, 1,1,1, 1,1,1};

   float* re_direct = conv_direct(matrix_d, matrix_g);
   printMatrix("direct", re_direct, 2,2);

   float* re_winograd = winograd_m2r3(matrix_d, matrix_g);
   printMatrix("winograd", re_winograd, 2,2);
    
   return 0;
}

