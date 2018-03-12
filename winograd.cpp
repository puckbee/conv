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

   printMatrix("UUUUU", matrix_U, 4,4);
   printMatrix("VVVVV", matrix_V, 4,4);


   float* matrix_UV = matrixDotProduct(matrix_U, matrix_V, 4,4);

   float* re = matrixMul(matrixMul(matrix_At, matrix_UV, 2,4,4), matrix_A, 2,4,2);

   return re;  
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

float* winograd(float* matrix_d, float* matrix_g, int d_h, int d_w)
{
   int g_w = 3;
   int g_h = 3;

   int tile_w = 4;
   int tile_h = 4;

   int re_w = d_w - g_w + 1;
   int re_h = d_h - g_h + 1;
   float* re = (float*) malloc(sizeof(float) * re_w * re_h);

   int numTile_w = (d_w - tile_w) / (g_w - 1) + 1;
   int numTile_h = (d_w - tile_h) / (g_h - 1) + 1;
 
   float* winoTile = (float*) malloc(sizeof(float) * tile_w * tile_h);

   int i,j,k,l, s,t;

   for(i=0; i<numTile_h; i++)
      for(j=0; j<numTile_w; j++)
      {
         int start_h = i * (tile_h - (g_h -1));
         int start_w = j * (tile_w - (g_w -1));
         for(k=0; k<tile_h; k++)
            for(l=0; l<tile_w; l++)
               winoTile[k*tile_w + l] = matrix_d[(start_h + k) * d_w + start_w + l];

         float* re_Tile = winograd_m2r3(winoTile, matrix_g);
         
         for(s=0; s<2; s++)
           for(t=0; t<2; t++)
               re[(i*2 + s)* numTile_w * 2 + j * 2 + t] = re_Tile[s*2+t];
      }
         
    return re;
}

int main(int argc, char** argv)
{
//   float matrix_d[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
//   float matrix_g[9]  = {1,1,1, 1,1,1, 1,1,1};

   float matrix_d[64] = {1,2,3,4,5,6,7,8, 5,6,7,8,9,10,11,12, 9,10,11,12,13,14,15,16, 13,14,15,16,17,18,19,20, 18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33, 34,35,36,37,38,39,40,41, 42,43,44,45,46,47,48,49};
   float matrix_g[9]  = {1,2,3, 4,5,6, 7,8,9};

   float* re_direct = conv_direct(matrix_d, matrix_g, 8,8);
   printMatrix("direct", re_direct, 6,6);

   float* re_winograd = winograd(matrix_d, matrix_g, 8,8);
   printMatrix("winograd", re_winograd, 6,6);

//   float* re_winograd = winograd_m2r3(matrix_d, matrix_g);
//   printMatrix("winograd", re_winograd, 2,2);

   
    
   return 0;
}




