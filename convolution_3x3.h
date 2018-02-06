// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

namespace ncnn{

static void conv3x3s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
{

//	printf(" Entering in conv_3x3 \n");


    int w = bottom_blob.w;
	int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

/*
	printf(" rows = %d, cols = %d, inch = %d \n", w, h, inch);

	for(int cc=0; cc<inch; cc++)
	{	
		printf("\nChannel %d:\n",cc);
		for(int ii=0; ii<h; ii++)
			for(int jj=0; jj<w; jj++)
				printf("%f ", (float)bottom_blob.channel(cc)[ii*w + jj]);
	}
*/	

    printf(" the first row of feature map\n");

    for(int cc=0; cc<inch; cc++)
        for(int ii =0; ii<3; ii++)
            for(int jj =0; jj<3; jj++)
            printf("%f ", bottom_blob.channel(cc)[ii*bottom_blob.w + jj]);

    printf("\n");


    const float* kernel = _kernel;
    const float* bias = _bias;

    float sum=0;
    for(int cc=0; cc<inch; cc++)
        for(int ii=0; ii<3; ii++)
            for(int jj=0; jj<3; jj++)
            {
                sum+= bottom_blob.channel(cc)[ii*bottom_blob.w +jj] * kernel[cc*9+ii*3+jj];
                printf(" %f x %f ;", bottom_blob.channel(cc)[ii*bottom_blob.w+jj], kernel[cc*9+ii*3+jj] );
            }

    printf("\n Sum[0] of conv = %f, bias = %f\n", sum, bias[0]);

    #pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*9  + q*9;
		/*	
			printf(" \n");
			for (int q9=0; q9<9; q9++) printf("%f ", kernel0[q9]);
			printf(" \n");
		*/
            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            int i = 0;

            for (; i+1 < outh; i+=2)
            {

                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {
                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;

                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

        }
    }

}




inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}


static void conv3x3s1_gemm_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
{

    int width = bottom_blob.w;
    int height = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

	const float* data_im = bottom_blob;
    const float* kernel = _kernel;
    const float* bias = _bias;

	int pad_h = 1;
	int pad_w = 1;
	int stride_h = 2;
	int stride_w = 2;
	int dilation_h = 1;
	int dilation_w = 1;
	int kernel_h = 3;
	int kernel_w = 3;

	const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

	float* data_col = (float*) malloc(sizeof(float) * kernel_w * kernel_h * inch * output_h * output_w);

//	im2col_cpu(data_im, inch, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, data_col);


}

}

