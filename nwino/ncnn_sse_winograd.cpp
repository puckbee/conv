static void conv3x3s1_winograd64_transform_kernel_neon(const Mat& kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8*8, inch, outch);

    const float ktm[8][3] = {
        {   1.0f,     0.0f,     0.0f},
        {-2.0f/9,  -2.0f/9,  -2.0f/9},
        {-2.0f/9,   2.0f/9,  -2.0f/9},
        {1.0f/90,  1.0f/45,  2.0f/45},
        {1.0f/90, -1.0f/45,  2.0f/45},
        {1.0f/45,  1.0f/90, 1.0f/180},
        {1.0f/45, -1.0f/90, 1.0f/180},
        {   0.0f,     0.0f,     1.0f}
    };

    #pragma omp parallel for
    for (int p = 0; p<outch; p++)
    {
        for (int q = 0; q<inch; q++)
        {
            const float* kernel_ncnn_0 = kernel.data + p*inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel, transposed
            const float* k0 = kernel_ncnn_0;
            const float* k1 = kernel_ncnn_0 + 3;
            const float* k2 = kernel_ncnn_0 + 6;

            // h
			/*
            float tmp[8][3];
            for (int i=0; i<8; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // v
            for (int j=0; j<8; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i=0; i<8; i++)
                {
                    kernel_tm0[j*8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }*/
			__m256 kernel_res0,kernel_res1,kernel_res2,kernel_t[8],kernel_tmp[3];
			__m128 kernel_af_1,kernel_af_2,kernel_af_3,kernel_af_4,kernel_af_5,kernel_af_6;
			__m128 kernel_tmp1,kernel_tmp2,kernel_tmp3;
			__m128 kernel_0,kernel_1,kernel_2;
			__m128 kernel0,kernel1,kernel2;
			__m128 kernel_con_4 = _mm_set_ps(4,4,4,4);
			__m128 kernel_con_2 = _mm_set_ps(2,2,2,2);
			__m128 kernel_con_2_9 = _mm_set_ps(-2.0f/9,-2.0f/9,-2.0f/9,-2.0f/9);
			__m128 kernel_con_1_90 = _mm_set_ps( 1.0f/90, 1.0f/90, 1.0f/90, 1.0f/90);
			__m128 kernel_con_1_180= _mm_set_ps( 1.0f/180, 1.0f/180, 1.0f/180, 1.0f/180);
		
			__m256 kernel_con1_4 = _mm256_set_ps(4,4,4,4,4,4,4,4);
			__m256 kernel_con1_2 = _mm256_set_ps(2,2,2,2,2,2,2,2);
			__m256 kernel_con1_2_9 = _mm256_set_ps(-2.0f/9,-2.0f/9,-2.0f/9,-2.0f/9,-2.0f/9,-2.0f/9,-2.0f/9,-2.0f/9);
			__m256 kernel_con1_1_90 = _mm256_set_ps(1.0f/90, 1.0f/90, 1.0f/90, 1.0f/90, 1.0f/90, 1.0f/90, 1.0f/90, 1.0f/90);
			__m256 kernel_con1_1_180= _mm256_set_ps(1.0f/180, 1.0f/180, 1.0f/180, 1.0f/180, 1.0f/180, 1.0f/180, 1.0f/180, 1.0f/180);
			
			kernel_0=_mm_loadu_ps(k0);
			kernel_1=_mm_loadu_ps(k1);
			kernel_2=_mm_loadu_ps(k2);
			kernel_tmp2=_mm_mul_ps(kernel_con_4,kernel_2);
			kernel_tmp3=_mm_mul_ps(kernel_con_2,kernel_1);
	
			kernel_tmp1=_mm_add_ps(kernel_0,kernel_2);
			kernel_af_1=_mm_add_ps(kernel_tmp1,kernel_1);
			kernel_af_1=_mm_mul_ps(kernel_af_1,kernel_con_2_9);
			kernel_af_2=_mm_sub_ps(kernel_tmp1,kernel_1);
			kernel_af_2=_mm_mul_ps(kernel_af_2,kernel_con_2_9);
		
			kernel_tmp1=_mm_add_ps(kernel_0,kernel_tmp2);
			kernel_af_3=_mm_add_ps(kernel_tmp1,kernel_tmp3);
			kernel_af_3=_mm_mul_ps(kernel_af_3,kernel_con_1_90);
			
			kernel_tmp2=_mm_mul_ps(kernel_0,kernel_con_4);
	
			kernel_af_4=_mm_sub_ps(kernel_tmp1,kernel_tmp3);
			kernel_af_4=_mm_mul_ps(kernel_af_4,kernel_con_1_90);
			
			kernel_tmp1=_mm_add_ps(kernel_2,kernel_tmp2);
			kernel_af_5=_mm_add_ps(kernel_tmp1,kernel_tmp3);
			kernel_af_5=_mm_mul_ps(kernel_af_5,kernel_con_1_180);
	
			kernel_af_6=_mm_sub_ps(kernel_tmp1,kernel_tmp3);
			kernel_af_6=_mm_mul_ps(kernel_af_6,kernel_con_1_180);
	
			kernel_res0=_mm256_set_ps(kernel_2[0],kernel_af_6[0],kernel_af_5[0],kernel_af_4[0],kernel_af_3[0],kernel_af_2[0],kernel_af_1[0],kernel_0[0]);
			kernel_res1=_mm256_set_ps(kernel_2[1],kernel_af_6[1],kernel_af_5[1],kernel_af_4[1],kernel_af_3[1],kernel_af_2[1],kernel_af_1[1],kernel_0[1]);
			kernel_res2=_mm256_set_ps(kernel_2[2],kernel_af_6[2],kernel_af_5[2],kernel_af_4[2],kernel_af_3[2],kernel_af_2[2],kernel_af_1[2],kernel_0[2]);
	
	
			kernel_tmp[2]=_mm256_mul_ps(kernel_con1_4,kernel_res2);
			kernel_tmp[3]=_mm256_mul_ps(kernel_con1_2,kernel_res1);
	
			kernel_tmp[1]=_mm256_add_ps(kernel_res0,kernel_res2);
			kernel_t[1]=_mm256_add_ps(kernel_tmp[1],kernel_res1);
			kernel_t[1]=_mm256_mul_ps(kernel_t[1],kernel_con1_2_9);
			kernel_t[2]=_mm256_sub_ps(kernel_tmp[1],kernel_res1);
			kernel_t[2]=_mm256_mul_ps(kernel_t[2],kernel_con1_2_9);
		
			kernel_tmp[1]=_mm256_add_ps(kernel_res0,kernel_tmp[2]);
			kernel_t[3]=_mm256_add_ps(kernel_tmp[1],kernel_tmp[3]);
			kernel_t[3]=_mm256_mul_ps(kernel_t[3],kernel_con1_1_90);
			
			kernel_tmp[2]=_mm256_mul_ps(kernel_res0,kernel_con1_4);
	
			kernel_t[4]=_mm256_sub_ps(kernel_tmp[1],kernel_tmp[3]);
			kernel_t[4]=_mm256_mul_ps(kernel_t[4],kernel_con1_1_90);
			
			kernel_tmp[1]=_mm256_add_ps(kernel_res2,kernel_tmp[2]);
			kernel_t[5]=_mm256_add_ps(kernel_tmp[1],kernel_tmp[3]);
			kernel_t[5]=_mm256_mul_ps(kernel_t[5],kernel_con1_1_180);
	
			kernel_t[6]=_mm256_sub_ps(kernel_tmp[1],kernel_tmp[3]);
			kernel_t[6]=_mm256_mul_ps(kernel_t[6],kernel_con1_1_180);
	
			_mm256_storeu_ps(kernel_tm0,kernel_res0);
			_mm256_storeu_ps(kernel_tm0 + 8,kernel_t[1]);
			_mm256_storeu_ps(kernel_tm0 + 8*2,kernel_t[2]);
			_mm256_storeu_ps(kernel_tm0 + 8*3,kernel_t[3]);
			_mm256_storeu_ps(kernel_tm0 + 8*4,kernel_t[4]);
			_mm256_storeu_ps(kernel_tm0 + 8*5,kernel_t[5]);
			_mm256_storeu_ps(kernel_tm0 + 8*6,kernel_t[6]);
			_mm256_storeu_ps(kernel_tm0 + 8*7,kernel_res2);

        }
    }
}



static void conv3x3s1_winograd64_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f);

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
		__m256 ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7;
		__m256 tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7;
		__m256 tmp8,tmp9,tmp10;
		__m256 con_5_25,con_4_25,con_0_25,con_1_25,con_5,con_4,con_2;
	
		con_5_25 = _mm256_set_ps(5.25,5.25,5.25,5.25,5.25,5.25,5.25,5.25);
		con_4_25 = _mm256_set_ps(4.25,4.25,4.25,4.25,4.25,4.25,4.25,4.25);
		con_0_25 = _mm256_set_ps(0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25);
		con_1_25 = _mm256_set_ps(1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25);
		con_5    = _mm256_set_ps(5,5,5,5,5,5,5,5);
		con_4    = _mm256_set_ps(4,4,4,4,4,4,4,4);
		con_2    = _mm256_set_ps(2,2,2,2,2,2,2,2);
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        bottom_blob_tm.create(8*8, w_tm/8 * h_tm/8, inch);

        #pragma omp parallel for
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
                    const float* r0 = img0.row(i * 6) + j * 6;
                    float* r0_tm = img0_tm.row(i * w_tm/8 + j);
					/*
                    // TODO neon optimize
                    for (int m=0; m<8; m++)
                    {
                        tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25;
                        tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25;

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25);

                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25 - r0[4] * 1.25);
                        float tmp34b = (r0[1] * 0.5 - r0[3] * 2.5 + r0[5] * 2);

                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25) * 4);
                        float tmp56b = (r0[1] * 2 - r0[3] * 2.5 + r0[5] * 0.5);

                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        r0 += w;
                    }

                    for (int m=0; m<8; m++)
                    {
                        const float* tmp0 = tmp[m];

                        r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
                        r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
                        float tmp12b = (tmp0[1] - tmp0[3] * 4.25 + tmp0[5]);

                        r0_tm[1] = tmp12a + tmp12b;
                        r0_tm[2] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
                        float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        r0_tm[3] = tmp34a + tmp34b;
                        r0_tm[4] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
                        float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        r0_tm[5] = tmp56a + tmp56b;
                        r0_tm[6] = tmp56a - tmp56b;

                        r0_tm += 8;
                    }
                    */
					tmp0 = _mm256_loadu_ps(r0);
					tmp1 = _mm256_loadu_ps(r0 + w);
					tmp2 = _mm256_loadu_ps(r0 + w * 2);
					tmp3 = _mm256_loadu_ps(r0 + w * 3);
					tmp4 = _mm256_loadu_ps(r0 + w * 4);
					tmp5 = _mm256_loadu_ps(r0 + w * 5);
					tmp6 = _mm256_loadu_ps(r0 + w * 6);
					tmp7 = _mm256_loadu_ps(r0 + w * 7);
				//res=BT*d
					tmp10 = _mm256_fnmadd_ps(con_4_25,tmp4,tmp2);
					tmp8 = _mm256_sub_ps(tmp0,tmp6);
					tmp9 = _mm256_sub_ps(tmp4,tmp2);
					ymm0 = _mm256_fmadd_ps(con_5_25,tmp9,tmp8);
					
					tmp8 = _mm256_fnmadd_ps(con_4_25,tmp3,tmp5);
					tmp0 = _mm256_add_ps(tmp10,tmp6);
					tmp10= _mm256_fmadd_ps(con_0_25,tmp2,tmp6);
					tmp9 = _mm256_add_ps(tmp8,tmp1);
					tmp8 = _mm256_fmadd_ps(con_0_25,tmp1,tmp5);
					tmp10= _mm256_fnmadd_ps(con_1_25,tmp4,tmp10);
					ymm1 = _mm256_add_ps(tmp0,tmp9);
					tmp8 = _mm256_fnmadd_ps(con_1_25,tmp3,tmp8);
					ymm2 = _mm256_sub_ps(tmp0,tmp9);
					
					tmp9 = _mm256_fnmadd_ps(con_5,tmp4,tmp6);
					tmp0 = _mm256_fmadd_ps(con_0_25,tmp5,tmp1);
					ymm3 = _mm256_fmadd_ps(con_2,tmp8,tmp10);
					ymm4 = _mm256_fnmadd_ps(con_2,tmp8,tmp10);
				
					tmp8 = _mm256_fmadd_ps(con_4,tmp2,tmp9);
					tmp10= _mm256_fnmadd_ps(con_1_25,tmp3,tmp0);
				
					tmp0 = _mm256_sub_ps(tmp7,tmp1);
					tmp9 = _mm256_sub_ps(tmp3,tmp5);
					ymm7 = _mm256_fmadd_ps(con_5_25,tmp9,tmp0);
				
					ymm5 = _mm256_fmadd_ps(con_2,tmp10,tmp8);
					ymm6 = _mm256_fnmadd_ps(con_2,tmp10,tmp8);
				
				//BT*d*B
				//test
				
					tmp0 = _mm256_set_ps(ymm7[0],ymm6[0],ymm5[0],ymm4[0],ymm3[0],ymm2[0],ymm1[0],ymm0[0]);
					tmp1 = _mm256_set_ps(ymm7[1],ymm6[1],ymm5[1],ymm4[1],ymm3[1],ymm2[1],ymm1[1],ymm0[1]);
					tmp2 = _mm256_set_ps(ymm7[2],ymm6[2],ymm5[2],ymm4[2],ymm3[2],ymm2[2],ymm1[2],ymm0[2]);
					tmp3 = _mm256_set_ps(ymm7[3],ymm6[3],ymm5[3],ymm4[3],ymm3[3],ymm2[3],ymm1[3],ymm0[3]);
					tmp4 = _mm256_set_ps(ymm7[4],ymm6[4],ymm5[4],ymm4[4],ymm3[4],ymm2[4],ymm1[4],ymm0[4]);
					tmp5 = _mm256_set_ps(ymm7[5],ymm6[5],ymm5[5],ymm4[5],ymm3[5],ymm2[5],ymm1[5],ymm0[5]);
					tmp6 = _mm256_set_ps(ymm7[6],ymm6[6],ymm5[6],ymm4[6],ymm3[6],ymm2[6],ymm1[6],ymm0[6]);
					tmp7 = _mm256_set_ps(ymm7[7],ymm6[7],ymm5[7],ymm4[7],ymm3[7],ymm2[7],ymm1[7],ymm0[7]);
				
				
					tmp10 = _mm256_fnmadd_ps(con_4_25,tmp4,tmp2);
					tmp8 = _mm256_sub_ps(tmp0,tmp6);
					tmp9 = _mm256_sub_ps(tmp4,tmp2);
					ymm0 = _mm256_fmadd_ps(con_5_25,tmp9,tmp8);
					
					tmp8 = _mm256_fnmadd_ps(con_4_25,tmp3,tmp5);
					tmp0 = _mm256_add_ps(tmp10,tmp6);
					tmp10= _mm256_fmadd_ps(con_0_25,tmp2,tmp6);
					tmp9 = _mm256_add_ps(tmp8,tmp1);
					tmp8 = _mm256_fmadd_ps(con_0_25,tmp1,tmp5);
					tmp10= _mm256_fnmadd_ps(con_1_25,tmp4,tmp10);
					ymm1 = _mm256_add_ps(tmp0,tmp9);
					tmp8 = _mm256_fnmadd_ps(con_1_25,tmp3,tmp8);
					ymm2 = _mm256_sub_ps(tmp0,tmp9);
					
					tmp9 = _mm256_fnmadd_ps(con_5,tmp4,tmp6);
					tmp0 = _mm256_fmadd_ps(con_0_25,tmp5,tmp1);
					ymm3 = _mm256_fmadd_ps(con_2,tmp8,tmp10);
					ymm4 = _mm256_fnmadd_ps(con_2,tmp8,tmp10);
				
					tmp8 = _mm256_fmadd_ps(con_4,tmp2,tmp9);
					tmp10= _mm256_fnmadd_ps(con_1_25,tmp3,tmp0);
				
					tmp0 = _mm256_sub_ps(tmp7,tmp1);
					tmp9 = _mm256_sub_ps(tmp3,tmp5);
					ymm7 = _mm256_fmadd_ps(con_5_25,tmp9,tmp0);
				
					ymm5 = _mm256_fmadd_ps(con_2,tmp10,tmp8);
					ymm6 = _mm256_fnmadd_ps(con_2,tmp10,tmp8);
				
					_mm256_store_ps(r0_tm,ymm0);
					_mm256_store_ps(r0_tm+8,ymm1);
					_mm256_store_ps(r0_tm+8*2,ymm2);
					_mm256_store_ps(r0_tm+8*3,ymm3);
					_mm256_store_ps(r0_tm+8*4,ymm4);
					_mm256_store_ps(r0_tm+8*5,ymm5);
					_mm256_store_ps(r0_tm+8*6,ymm6);
					_mm256_store_ps(r0_tm+8*7,ymm7);
                }
            }
        }

    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        top_blob_tm.create(8*8, w_tm/8 * h_tm/8, outch);

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);
            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p+1);
            const Mat kernel2_tm = kernel_tm.channel(p+2);
            const Mat kernel3_tm = kernel_tm.channel(p+3);

            out0_tm.fill(0.f);
            out1_tm.fill(0.f);
            out2_tm.fill(0.f);
            out3_tm.fill(0.f);

            int q = 0;
            for (; q+3<inch; q+=4)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);
                const float* r2 = bottom_blob_tm.channel(q+2);
                const float* r3 = bottom_blob_tm.channel(q+3);

                const float* k00 = kernel0_tm.row(q);
                const float* k10 = kernel1_tm.row(q);
                const float* k20 = kernel2_tm.row(q);
                const float* k30 = kernel3_tm.row(q);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

                // tile
                for (int i=0; i<h_tm/8 * w_tm/8; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    for (int m=0; m+7<64; m+=8)
                    {
                        float32x4_t _output0_tm = vld1q_f32(output0_tm);
                        float32x4_t _output1_tm = vld1q_f32(output1_tm);
                        float32x4_t _output2_tm = vld1q_f32(output2_tm);
                        float32x4_t _output3_tm = vld1q_f32(output3_tm);

                        float32x4_t _r0 = vld1q_f32(r0);
                        float32x4_t _r1 = vld1q_f32(r1);
                        float32x4_t _r2 = vld1q_f32(r2);
                        float32x4_t _r3 = vld1q_f32(r3);

                        float32x4_t _k00 = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k01 = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k02 = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k03 = vld1q_f32(k00);
                        k00 += 64;

                        k00 -= 64*4;

                        _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                        _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                        _output0_tm = vmlaq_f32(_output0_tm, _r2, _k02);
                        _output0_tm = vmlaq_f32(_output0_tm, _r3, _k03);

                        float32x4_t _k10 = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k11 = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k12 = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k13 = vld1q_f32(k10);
                        k10 += 64;

                        k10 -= 64*4;

                        _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                        _output1_tm = vmlaq_f32(_output1_tm, _r1, _k11);
                        _output1_tm = vmlaq_f32(_output1_tm, _r2, _k12);
                        _output1_tm = vmlaq_f32(_output1_tm, _r3, _k13);

                        float32x4_t _k20 = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k21 = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k22 = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k23 = vld1q_f32(k20);
                        k20 += 64;

                        k20 -= 64*4;

                        _output2_tm = vmlaq_f32(_output2_tm, _r0, _k20);
                        _output2_tm = vmlaq_f32(_output2_tm, _r1, _k21);
                        _output2_tm = vmlaq_f32(_output2_tm, _r2, _k22);
                        _output2_tm = vmlaq_f32(_output2_tm, _r3, _k23);

                        float32x4_t _k30 = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k31 = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k32 = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k33 = vld1q_f32(k30);
                        k30 += 64;

                        k30 -= 64*4;

                        _output3_tm = vmlaq_f32(_output3_tm, _r0, _k30);
                        _output3_tm = vmlaq_f32(_output3_tm, _r1, _k31);
                        _output3_tm = vmlaq_f32(_output3_tm, _r2, _k32);
                        _output3_tm = vmlaq_f32(_output3_tm, _r3, _k33);

                        vst1q_f32(output0_tm, _output0_tm);
                        vst1q_f32(output1_tm, _output1_tm);
                        vst1q_f32(output2_tm, _output2_tm);
                        vst1q_f32(output3_tm, _output3_tm);

                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;

                        k00 += 4;
                        k10 += 4;
                        k20 += 4;
                        k30 += 4;

                        float32x4_t _output0_tmn = vld1q_f32(output0_tm);
                        float32x4_t _output1_tmn = vld1q_f32(output1_tm);
                        float32x4_t _output2_tmn = vld1q_f32(output2_tm);
                        float32x4_t _output3_tmn = vld1q_f32(output3_tm);

                        float32x4_t _r0n = vld1q_f32(r0);
                        float32x4_t _r1n = vld1q_f32(r1);
                        float32x4_t _r2n = vld1q_f32(r2);
                        float32x4_t _r3n = vld1q_f32(r3);

                        float32x4_t _k00n = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k01n = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k02n = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k03n = vld1q_f32(k00);
                        k00 += 64;

                        k00 -= 64*4;

                        _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r2n, _k02n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r3n, _k03n);

                        float32x4_t _k10n = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k11n = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k12n = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k13n = vld1q_f32(k10);
                        k10 += 64;

                        k10 -= 64*4;

                        _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);
                        _output1_tmn = vmlaq_f32(_output1_tmn, _r1n, _k11n);
                        _output1_tmn = vmlaq_f32(_output1_tmn, _r2n, _k12n);
                        _output1_tmn = vmlaq_f32(_output1_tmn, _r3n, _k13n);

                        float32x4_t _k20n = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k21n = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k22n = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k23n = vld1q_f32(k20);
                        k20 += 64;

                        k20 -= 64*4;

                        _output2_tmn = vmlaq_f32(_output2_tmn, _r0n, _k20n);
                        _output2_tmn = vmlaq_f32(_output2_tmn, _r1n, _k21n);
                        _output2_tmn = vmlaq_f32(_output2_tmn, _r2n, _k22n);
                        _output2_tmn = vmlaq_f32(_output2_tmn, _r3n, _k23n);

                        float32x4_t _k30n = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k31n = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k32n = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k33n = vld1q_f32(k30);
                        k30 += 64;

                        k30 -= 64*4;

                        _output3_tmn = vmlaq_f32(_output3_tmn, _r0n, _k30n);
                        _output3_tmn = vmlaq_f32(_output3_tmn, _r1n, _k31n);
                        _output3_tmn = vmlaq_f32(_output3_tmn, _r2n, _k32n);
                        _output3_tmn = vmlaq_f32(_output3_tmn, _r3n, _k33n);

                        vst1q_f32(output0_tm, _output0_tmn);
                        vst1q_f32(output1_tm, _output1_tmn);
                        vst1q_f32(output2_tm, _output2_tmn);
                        vst1q_f32(output3_tm, _output3_tmn);

                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;

                        k00 += 4;
                        k10 += 4;
                        k20 += 4;
                        k30 += 4;
                    }
#else // __aarch64__
                    asm volatile(
                        "mov        r4, #8              \n"

                        "pld        [%0, #256]          \n"  //预取
                        "vld1.f32   {d16-d19}, [%0 :128]\n"//q8 q9 = _output0_tm

                        "0:                             \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d0-d3}, [%4 :128]! \n"//q0 q1 = _r0

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]\n"//q10 q11 = _k00
                        "add        %8, %8, #256        \n"

                        "vmla.f32   q8, q0, q10         \n"
                        "vmla.f32   q9, q1, q11         \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d24-d27}, [%1 :128]\n"//q12 q13 = _output1_tm

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d28-d31}, [%9 :128]\n"//q14 q15 = _k10
                        "add        %9, %9, #256        \n"

                        "vmla.f32   q12, q0, q14        \n"
                        "vmla.f32   q13, q1, q15        \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"//q2 q3 = _r1

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]\n"//q10 q11 = _k01
                        "add        %8, %8, #256        \n"

                        "vmla.f32   q8, q2, q10         \n"
                        "vmla.f32   q9, q3, q11         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d28-d31}, [%9 :128]\n"//q14 q15 = _k11
                        "add        %9, %9, #256        \n"

                        "vmla.f32   q12, q2, q14        \n"
                        "vmla.f32   q13, q3, q15        \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d8-d11}, [%6 :128]!\n"//q4 q5 = _r2

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]\n"//q10 q11 = _k02
                        "add        %8, %8, #256        \n"

                        "vmla.f32   q8, q4, q10         \n"
                        "vmla.f32   q9, q5, q11         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d28-d31}, [%9 :128]\n"//q14 q15 = _k12
                        "add        %9, %9, #256        \n"

                        "vmla.f32   q12, q4, q14        \n"
                        "vmla.f32   q13, q5, q15        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d12-d15}, [%7 :128]!\n"//q6 q7 = _r3

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]\n"//q10 q11 = _k03
                        "sub        %8, %8, #736        \n"

                        "vmla.f32   q8, q6, q10         \n"
                        "vmla.f32   q9, q7, q11         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d28-d31}, [%9 :128]\n"//q14 q15 = _k13
                        "sub        %9, %9, #736        \n"

                        "vmla.f32   q12, q6, q14        \n"
                        "vmla.f32   q13, q7, q15        \n"

                        "vst1.f32   {d16-d19}, [%0 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]\n"//q8 q9 = _output2_tm

                        "pld        [%10, #256]         \n"
                        "vld1.f32   {d20-d23}, [%10 :128]\n"//q10 q11 = _k20
                        "add        %10, %10, #256      \n"

                        "vmla.f32   q8, q0, q10         \n"
                        "vmla.f32   q9, q1, q11         \n"

                        "vst1.f32   {d24-d27}, [%1 :128]!\n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d24-d27}, [%3 :128]\n"//q12 q13 = _output3_tm

                        "pld        [%11, #256]         \n"
                        "vld1.f32   {d28-d31}, [%11 :128]\n"//q14 q15 = _k30
                        "add        %11, %11, #256      \n"

                        "vmla.f32   q12, q0, q14        \n"
                        "vmla.f32   q13, q1, q15        \n"

                        "pld        [%10, #256]         \n"
                        "vld1.f32   {d20-d23}, [%10 :128]\n"//q10 q11 = _k21
                        "add        %10, %10, #256      \n"

                        "vmla.f32   q8, q2, q10         \n"
                        "vmla.f32   q9, q3, q11         \n"

                        "pld        [%11, #256]         \n"
                        "vld1.f32   {d28-d31}, [%11 :128]\n"//q14 q15 = _k31
                        "add        %11, %11, #256      \n"

                        "vmla.f32   q12, q2, q14        \n"
                        "vmla.f32   q13, q3, q15        \n"

                        "pld        [%10, #256]         \n"
                        "vld1.f32   {d20-d23}, [%10 :128]\n"//q10 q11 = _k22
                        "add        %10, %10, #256      \n"

                        "vmla.f32   q8, q4, q10         \n"
                        "vmla.f32   q9, q5, q11         \n"

                        "pld        [%11, #256]         \n"
                        "vld1.f32   {d28-d31}, [%11 :128]\n"//q14 q15 = _k32
                        "add        %11, %11, #256      \n"

                        "vmla.f32   q12, q4, q14        \n"
                        "vmla.f32   q13, q5, q15        \n"

                        "pld        [%10, #256]         \n"
                        "vld1.f32   {d20-d23}, [%10 :128]\n"//q10 q11 = _k23
                        "sub        %10, %10, #736      \n"

                        "vmla.f32   q8, q6, q10         \n"
                        "vmla.f32   q9, q7, q11         \n"

                        "pld        [%11, #256]         \n"
                        "vld1.f32   {d28-d31}, [%11 :128]\n"//q14 q15 = _k33
                        "sub        %11, %11, #736      \n"

                        "vmla.f32   q12, q6, q14        \n"
                        "vmla.f32   q13, q7, q15        \n"

                        "vst1.f32   {d16-d19}, [%2 :128]!\n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d16-d19}, [%0 :128]\n"//q8 q9 = _output0_tm

                        "subs       r4, r4, #1          \n"

                        "vst1.f32   {d24-d27}, [%3 :128]!\n"

                        "bne        0b                  \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(r0),         // %4
                          "=r"(r1),         // %5
                          "=r"(r2),         // %6
                          "=r"(r3),         // %7
                          "=r"(k00),        // %8
                          "=r"(k10),        // %9
                          "=r"(k20),        // %10
                          "=r"(k30)         // %11
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(r0),
                          "5"(r1),
                          "6"(r2),
                          "7"(r3),
                          "8"(k00),
                          "9"(k10),
                          "10"(k20),
                          "11"(k30)
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__

                    k00 -= 64;
                    k10 -= 64;
                    k20 -= 64;
                    k30 -= 64;
#else
                    for (int m=0; m<64; m++)
                    {
                        output0_tm[m] += r0[m] * k00[m];
                        k00 += 64;
                        output0_tm[m] += r1[m] * k00[m];
                        k00 += 64;
                        output0_tm[m] += r2[m] * k00[m];
                        k00 += 64;
                        output0_tm[m] += r3[m] * k00[m];
                        k00 += 64;

                        k00 -= 64 * 4;

                        output1_tm[m] += r0[m] * k10[m];
                        k10 += 64;
                        output1_tm[m] += r1[m] * k10[m];
                        k10 += 64;
                        output1_tm[m] += r2[m] * k10[m];
                        k10 += 64;
                        output1_tm[m] += r3[m] * k10[m];
                        k10 += 64;

                        k10 -= 64 * 4;

                        output2_tm[m] += r0[m] * k20[m];
                        k20 += 64;
                        output2_tm[m] += r1[m] * k20[m];
                        k20 += 64;
                        output2_tm[m] += r2[m] * k20[m];
                        k20 += 64;
                        output2_tm[m] += r3[m] * k20[m];
                        k20 += 64;

                        k20 -= 64 * 4;

                        output3_tm[m] += r0[m] * k30[m];
                        k30 += 64;
                        output3_tm[m] += r1[m] * k30[m];
                        k30 += 64;
                        output3_tm[m] += r2[m] * k30[m];
                        k30 += 64;
                        output3_tm[m] += r3[m] * k30[m];
                        k30 += 64;

                        k30 -= 64 * 4;
                    }

                    r0 += 64;
                    r1 += 64;
                    r2 += 64;
                    r3 += 64;
                    output0_tm += 64;
                    output1_tm += 64;
                    output2_tm += 64;
                    output3_tm += 64;
#endif // __ARM_NEON
                }
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                const float* k0 = kernel0_tm.row(q);
                const float* k1 = kernel1_tm.row(q);
                const float* k2 = kernel2_tm.row(q);
                const float* k3 = kernel3_tm.row(q);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

                // tile
                for (int i=0; i<h_tm/8 * w_tm/8; i++)
                {
                    // TODO neon optimize
                    for (int m=0; m<64; m++)
                    {
                        output0_tm[m] += r0[m] * k0[m];
                        output1_tm[m] += r0[m] * k1[m];
                        output2_tm[m] += r0[m] * k2[m];
                        output3_tm[m] += r0[m] * k3[m];
                    }

                    r0 += 64;
                    output0_tm += 64;
                    output1_tm += 64;
                    output2_tm += 64;
                    output3_tm += 64;
                }

            }
        }

        #pragma omp parallel for
        for (int p=remain_outch_start; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            out0_tm.fill(0.f);

            int q = 0;
            for (; q+3<inch; q+=4)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);
                const float* r2 = bottom_blob_tm.channel(q+2);
                const float* r3 = bottom_blob_tm.channel(q+3);

                const float* k0 = kernel0_tm.row(q);
                const float* k1 = kernel0_tm.row(q+1);
                const float* k2 = kernel0_tm.row(q+2);
                const float* k3 = kernel0_tm.row(q+3);

                float* output0_tm = out0_tm;

                // tile
                for (int i=0; i<h_tm/8 * w_tm/8; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    for (int m=0; m+7<64; m+=8)
                    {
                        float32x4_t _output0_tm = vld1q_f32(output0_tm);

                        float32x4_t _r0 = vld1q_f32(r0);
                        float32x4_t _r1 = vld1q_f32(r1);
                        float32x4_t _r2 = vld1q_f32(r2);
                        float32x4_t _r3 = vld1q_f32(r3);

                        float32x4_t _k0 = vld1q_f32(k0);
                        float32x4_t _k1 = vld1q_f32(k1);
                        float32x4_t _k2 = vld1q_f32(k2);
                        float32x4_t _k3 = vld1q_f32(k3);

                        _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0);
                        _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1);
                        _output0_tm = vmlaq_f32(_output0_tm, _r2, _k2);
                        _output0_tm = vmlaq_f32(_output0_tm, _r3, _k3);

                        vst1q_f32(output0_tm, _output0_tm);

                        output0_tm += 4;

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;

                        k0 += 4;
                        k1 += 4;
                        k2 += 4;
                        k3 += 4;

                        float32x4_t _output0_tmn = vld1q_f32(output0_tm);

                        float32x4_t _r0n = vld1q_f32(r0);
                        float32x4_t _r1n = vld1q_f32(r1);
                        float32x4_t _r2n = vld1q_f32(r2);
                        float32x4_t _r3n = vld1q_f32(r3);

                        float32x4_t _k0n = vld1q_f32(k0);
                        float32x4_t _k1n = vld1q_f32(k1);
                        float32x4_t _k2n = vld1q_f32(k2);
                        float32x4_t _k3n = vld1q_f32(k3);

                        _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r2n, _k2n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r3n, _k3n);

                        vst1q_f32(output0_tm, _output0_tmn);

                        output0_tm += 4;

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;

                        k0 += 4;
                        k1 += 4;
                        k2 += 4;
                        k3 += 4;
                    }
#else
                    asm volatile(
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "mov        r4, %0              \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128]!\n"//q12 q13 = output0_tm

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128]!\n"//q14 q15 = output0_tm
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "vst1.f32   {d24-d27}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128]!\n"//q12 q13 = output0_tm
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "vst1.f32   {d28-d31}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128]!\n"//q14 q15 = output0_tm
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "vst1.f32   {d24-d27}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128]!\n"//q12 q13 = output0_tm
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "vst1.f32   {d28-d31}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128]!\n"//q14 q15 = output0_tm
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "vst1.f32   {d24-d27}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128]!\n"//q12 q13 = output0_tm
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "vst1.f32   {d28-d31}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128]!\n"//q14 q15 = output0_tm
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "vst1.f32   {d24-d27}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "vst1.f32   {d28-d31}, [r4 :128]!\n"

                        : "=r"(output0_tm), // %0
                          "=r"(r0),         // %1
                          "=r"(r1),         // %2
                          "=r"(r2),         // %3
                          "=r"(r3),         // %4
                          "=r"(k0),         // %5
                          "=r"(k1),         // %6
                          "=r"(k2),         // %7
                          "=r"(k3)          // %8
                        : "0"(output0_tm),
                          "1"(r0),
                          "2"(r1),
                          "3"(r2),
                          "4"(r3),
                          "5"(k0),
                          "6"(k1),
                          "7"(k2),
                          "8"(k3)
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__

                    k0 -= 64;
                    k1 -= 64;
                    k2 -= 64;
                    k3 -= 64;
#else
                    for (int m=0; m<64; m++)
                    {
                        output0_tm[m] += r0[m] * k0[m];
                        output0_tm[m] += r1[m] * k1[m];
                        output0_tm[m] += r2[m] * k2[m];
                        output0_tm[m] += r3[m] * k3[m];
                    }

                    r0 += 64;
                    r1 += 64;
                    r2 += 64;
                    r3 += 64;
                    output0_tm += 64;
#endif // __ARM_NEON
                }
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                const float* k0 = kernel0_tm.row(q);

                float* output0_tm = out0_tm;

                // tile
                for (int i=0; i<h_tm/8 * w_tm/8; i++)
                {
                    // TODO neon optimize
                    for (int m=0; m<64; m++)
                    {
                        output0_tm[m] += r0[m] * k0[m];
                    }

                    r0 += 64;
                    output0_tm += 64;
                }

            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch);
    {
		__m256 ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7;
		__m256 tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmp10;
		__m256 con_5_25,con_4_25,con_0_25,con_1_25,con_5,con_4,con_2;
		__m256 kernel0,kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7;

		__m256 res0,res1,res2,res3,res4,res5,test6,test7;
		__m256 res_t0,res_t1,res_t2,res_t3,res_t4,res_t5,res_t6,res_t7;
		__m256 tmpt0,tmpt1,tmpt2,tmpt3,tmpt4,tmpt5,tmpt6,tmpt7;
		con_4    = _mm256_set_ps(4,4,4,4,4,4,4,4);
		con_2    = _mm256_set_ps(2,2,2,2,2,2,2,2);
		con_4_25= _mm256_set_ps(32,32,32,32,32,32,32,32);
		con_5_25= _mm256_set_ps(16,16,16,16,16,16,16,16);
		con_0_25= _mm256_set_ps(8,8,8,8,8,8,8,8);
        int w_tm = outw / 6 * 8;

        #pragma omp parallel for
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;

            float tmp[6][8];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
                    const float* output0_tm = out0_tm.row(i * w_tm/8 + j);
                    float* output0 = out0.row(i * 6) + j * 6;
					/*
                    // TODO neon optimize
                    for (int m=0; m<8; m++)
                    {
                        float tmp024a = output0_tm[1] + output0_tm[2];
                        float tmp135a = output0_tm[1] - output0_tm[2];

                        float tmp024b = output0_tm[3] + output0_tm[4];
                        float tmp135b = output0_tm[3] - output0_tm[4];

                        float tmp024c = output0_tm[5] + output0_tm[6];
                        float tmp135c = output0_tm[5] - output0_tm[6];

                        tmp[0][m] = output0_tm[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm += 8;
                    }

                    for (int m=0; m<6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw;
                    }
                    */
					
					
					tmpt0 = _mm256_loadu_ps(output0_tm);
					tmpt1 = _mm256_loadu_ps(output0_tm + 8);
					tmpt2 = _mm256_loadu_ps(output0_tm + 8*2);
					tmpt3 = _mm256_loadu_ps(output0_tm + 8*3);
					tmpt4 = _mm256_loadu_ps(output0_tm + 8*4);
					tmpt5 = _mm256_loadu_ps(output0_tm + 8*5);
					tmpt6 = _mm256_loadu_ps(output0_tm + 8*6);
					tmpt7 = _mm256_loadu_ps(output0_tm + 8*7 );
			
					tmp8 = _mm256_add_ps(tmpt1,tmpt2);
					tmp9 = _mm256_add_ps(tmpt5,tmpt6);
					tmp10= _mm256_add_ps(tmpt3,tmpt4);
					test6 = _mm256_sub_ps(tmpt5,tmpt6);
					test7 = _mm256_sub_ps(tmpt3,tmpt4);
			
					res0 = _mm256_fmadd_ps(con_4_25,tmp9,tmp10);
					res2 = _mm256_fmadd_ps(con_0_25,tmp9,tmp8);
					res4 = _mm256_fmadd_ps(con_2,tmp9,tmp8);
			
					res0 = _mm256_add_ps(res0,tmp8);
					res0 = _mm256_add_ps(res0,tmpt0);
				
					res2 = _mm256_fmadd_ps(con_4,tmp10,res2);
			
					res4 = _mm256_fmadd_ps(con_5_25,tmp10,res4);
			
					tmpt0 = _mm256_sub_ps(tmpt1,tmpt2);
			
					res1 = _mm256_fmadd_ps(con_2,test7,tmpt0);
					res3 = _mm256_fmadd_ps(con_0_25,test7,tmpt0);
					res5 = _mm256_fmadd_ps(con_4_25,test7,tmpt0);
				
					res1 = _mm256_fmadd_ps(con_5_25,test6,res1);
				
					res3 = _mm256_fmadd_ps(con_4,test6,res3);
				
					tmpt7 = _mm256_add_ps(test6,tmpt7);
					res5 = _mm256_add_ps(res5,tmpt7);
			
				//transpose
				
					tmpt0 = _mm256_set_ps(0,0,res5[0],res4[0],res3[0],res2[0],res1[0],res0[0]);
					tmpt1 = _mm256_set_ps(0,0,res5[1],res4[1],res3[1],res2[1],res1[1],res0[1]);
					tmpt2 = _mm256_set_ps(0,0,res5[2],res4[2],res3[2],res2[2],res1[2],res0[2]);
					tmpt3 = _mm256_set_ps(0,0,res5[3],res4[3],res3[3],res2[3],res1[3],res0[3]);
					tmpt4 = _mm256_set_ps(0,0,res5[4],res4[4],res3[4],res2[4],res1[4],res0[4]);
					tmpt5 = _mm256_set_ps(0,0,res5[5],res4[5],res3[5],res2[5],res1[5],res0[5]);
					tmpt6 = _mm256_set_ps(0,0,res5[6],res4[6],res3[6],res2[6],res1[6],res0[6]);
					tmpt7 = _mm256_set_ps(0,0,res5[7],res4[7],res3[7],res2[7],res1[7],res0[7]);
			
				
					tmp8 = _mm256_add_ps(tmpt1,tmpt2);
					tmp9 = _mm256_add_ps(tmpt5,tmpt6);
					tmp10= _mm256_add_ps(tmpt3,tmpt4);
					test6 = _mm256_sub_ps(tmpt5,tmpt6);
					test7 = _mm256_sub_ps(tmpt3,tmpt4);
			
					res0 = _mm256_fmadd_ps(con_4_25,tmp9,tmp10);
					res2 = _mm256_fmadd_ps(con_0_25,tmp9,tmp8);
					res4 = _mm256_fmadd_ps(con_2,tmp9,tmp8);
			
					res0 = _mm256_add_ps(res0,tmp8);
					res0 = _mm256_add_ps(res0,tmpt0);
				
					res2 = _mm256_fmadd_ps(con_4,tmp10,res2);
			
					res4 = _mm256_fmadd_ps(con_5_25,tmp10,res4);
			
					tmpt0 = _mm256_sub_ps(tmpt1,tmpt2);
			
					res1 = _mm256_fmadd_ps(con_2,test7,tmpt0);
					res3 = _mm256_fmadd_ps(con_0_25,test7,tmpt0);
					res5 = _mm256_fmadd_ps(con_4_25,test7,tmpt0);
				
					res1 = _mm256_fmadd_ps(con_5_25,test6,res1);
				
					res3 = _mm256_fmadd_ps(con_4,test6,res3);
				
					tmpt7 = _mm256_add_ps(test6,tmpt7);
					res5 = _mm256_add_ps(res5,tmpt7);
			
					
					/*
					_mm256_storeu_ps(output0,res0);
					_mm256_storeu_ps(output0 + outw*1,res1);
					_mm256_storeu_ps(output0 + outw*2,res2);
					_mm256_storeu_ps(output0 + outw*3,res3);
					_mm256_storeu_ps(output0 + outw*4,res4);
					_mm256_storeu_ps(output0 + outw*5,res5);
					*/
					//防止越界
					float *output00 =  output0 + outw*0;
					float *output1 =  output0 + outw*1;
					float *output2 =  output0 + outw*2;
					float *output3 =  output0 + outw*3;
					float *output4 =  output0 + outw*4;
					float *output5 =  output0 + outw*5;
					for(int z = 0;z < 6;z++)
					{
						*output00 = res0[z];
						*output1 = res1[z];
						*output2 = res2[z];
						*output3 = res3[z];
						*output4 = res4[z];
						*output5 = res5[z];
						output00++;
						output1++;
						output2++;
						output3++;
						output4++;
						output5++;
					}
					
					
                }
            }
        }
    }
    // END transform output

    // cut result pad
	//print(top_blob_bordered,"top_blob_bordered");
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w);
	//print(top_blob_bordered,"top_blob_bordered");
}