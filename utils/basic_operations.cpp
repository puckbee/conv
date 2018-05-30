#include "mat.h"
#include "basic_operations.h"

#include <time.h>
#include <sys/time.h>
#define MICRO_IN_SEC 1000000.00


double microtime(){
        int tv_sec,tv_usec;
        double time;
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}


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




void printBlob(string str, Mat blob)
{

    std::cout<<endl;
    std::cout<<" Blob "<<str;
    int nnz = 0;

	for(int cc=0; cc<blob.c; cc++)
		for(int ii=0; ii<blob.h; ii++)
			for(int jj=0; jj<blob.w; jj++)
			{
				printf(" %f", (float)blob.channel(cc)[ii*blob.w + jj]);
				if(blob.channel(cc)[ii*blob.w+jj] != 0)
					nnz++;
			}

    std::cout<<endl;
    std::cout<<" Density: "<<(float)nnz/blob.total()*100<<"%"<<endl;
    std::cout<<endl;
}


void readStr(string strLine, float* data)
{
	int p = 0;	

	char* kernelStr = (char*)malloc(sizeof(char) * strLine.length());
	strcpy(kernelStr, strLine.c_str());

	char* pStr;
	pStr = strtok(kernelStr," ");
	while(pStr)
	{
		data[p++] = atof(pStr);
		pStr = strtok(NULL," ");
	}

}


int read_data(Mat& bottom_blob, Mat& kernel_blob, Mat& xtop_blob, Mat& _bias_data, int &kernel_size, int& num_output)
{
	
	fstream fd;
	fd.open("1.txt");

	// read in the basic information
	string strLine;
	getline(fd,strLine);
	sscanf(strLine.c_str(), " %d %d %d, %d %d, %d %d %d\n", &bottom_blob.w, &bottom_blob.h, &bottom_blob.c, &kernel_size, &num_output, &xtop_blob.w, &xtop_blob.h, &xtop_blob.c);

	printf("Basic information of Conv(w,h,c):  bottom: %d, %d, %d; kernel: %d, %d, %d; xtop: %d, %d, %d \n \n", bottom_blob.w, bottom_blob.h, bottom_blob.c, kernel_size, kernel_size, num_output, xtop_blob.w, xtop_blob.h, xtop_blob.c);

	// create the basic data structures
	bottom_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.c);
	kernel_blob.create(kernel_size * kernel_size * bottom_blob.c * num_output);
	xtop_blob.create(xtop_blob.w, xtop_blob.h, xtop_blob.c);
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
	if(_bias_data.data == NULL)
	{
		printf(" xtop is null");
		return -1;
	}


// Start reading bottom	
	getline(fd,strLine);
	readStr(strLine, bottom_blob.data);

// Start reading kernel
	getline(fd,strLine);
	getline(fd,strLine);
	readStr(strLine, kernel_blob.data);

// Start reading bias
	getline(fd,strLine);
	readStr(strLine, _bias_data.data);

// Start reading reference top
	getline(fd,strLine);
	getline(fd,strLine);
	readStr(strLine, xtop_blob.data);
}

void generate_data(Mat& bottom_blob, Mat& kernel_blob, Mat& _bias_data, int in_channel, int out_channel, int bottom_w, int bottom_h, int kernel_size)
{

	bottom_blob.create(bottom_w, bottom_h, in_channel);
//	kernel_blob.create(kernel_size * kernel_size * in_channel * out_channel);
	kernel_blob.create(kernel_size, kernel_size, in_channel * out_channel);
	_bias_data.create(out_channel);

    srand( (unsigned)time( NULL ) );

    std::cout<<" bottom_blob.total is "<< bottom_blob.total()<<endl;

    for(int i=0; i<bottom_blob.c; i++)
        for(int j=0; j<bottom_blob.h * bottom_blob.w; j++)
        bottom_blob.channel(i)[j] = float(rand()%100)/1000;
        
    for(int i=0; i<kernel_blob.total(); i++)
        kernel_blob.data[i] = float(rand()%100)/1000;
    
    for(int i=0; i<_bias_data.total(); i++)
        _bias_data.data[i] = 0;
//        _bias_data.data[i] = float(rand()%100)/1000;

    std::cout<<" bottom "<< bottom_blob.total()<<"; kernel "<< kernel_blob.total()<<"; bias "<< _bias_data.total()<<endl;
}



void checkResults(float* x1, float* x2, int size)
{

	int res = 0;
	for(int i=0; i< size; i++)
	{
//        if(i==1 || i==0)
//			std::cout<<"Result["<<i<<"]="<< x1[i]<<";  Ref["<<i<<"]="<<x2[i]<<endl;
        
		if(fabs(x1[i] - x2[i]) > 0.000005*fabs(x1[i]))
//		if(fabs(x1[i] - x2[i]) > 0.5*fabs(x1[i]))
		{
			res++;
//            if((i%16==0 && i<100))
			std::cout<<"Result["<<i<<"]="<< x1[i]<<";  Ref["<<i<<"]="<<x2[i]<<endl;
		}
	}

	if(res==0)
		std::cout<<" Check Passed"<<endl;
	else
		std::cout<<" Warning! "<<res<<" out of "<< size <<" are wrong"<<endl;
}



void checkResults(Mat& x1, Mat& x2)
{

    int c = x1.c;
    int h = x1.h;
    int w = x1.w;

	int res = 0;
	for(int ic=0; ic< c; ic++)
	{
        float* x1data = x1.channel(ic);
        float* x2data = x2.channel(ic);

        for(int i=0; i< h * w; i ++)
        {       
		    if(fabs(x1data[i] - x2data[i]) > 0.000005*fabs(x1data[i]))
		    {
			   res++;
//             if((i%16==0 && i<100))
			   std::cout<<"Result["<<i<<"]="<< x1data[i]<<";  Ref["<<i<<"]="<<x2data[i]<<endl;
		    }
        }
	}

	if(res==0)
		std::cout<<" Check Passed"<<endl;
	else
		std::cout<<" Warning! "<<res<<" out of "<< c*w*h <<" are wrong"<<endl;
}



void checkResults(float* x1, Mat& x2)
{

    int c = x2.c;
    int h = x2.h;
    int w = x2.w;

	int res = 0;
	for(int ic=0; ic< c; ic++)
	{
        float* x1data = x1 + ic * h * w;
        float* x2data = x2.channel(ic);

        for(int i=0; i< h * w; i ++)
        {       
		    if(fabs(x1data[i] - x2data[i]) > 0.000005*fabs(x1data[i]))
		    {
			   res++;
//             if((i%16==0 && i<100))
			   std::cout<<"Result["<<i<<"]="<< x1data[i]<<";  Ref["<<i<<"]="<<x2data[i]<<endl;
		    }
        }
    }
	if(res==0)
		std::cout<<" Check Passed"<<endl;
	else
		std::cout<<" Warning! "<<res<<" out of "<< c*w*h <<" are wrong"<<endl;
}





























