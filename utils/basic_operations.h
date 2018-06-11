#include "mat.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


using namespace std;

double microtime();

void printBlob(string str, Mat blob);

void readStr(string strLine, float* data);


int read_data(Mat& bottom_blob, Mat& kernel_blob, Mat& xtop_blob, Mat& _bias_data, int &kernel_size, int& num_output);


void generate_data(Mat& bottom_blob, Mat& kernel_blob, Mat& _bias_data, int in_channel, int out_channel, int bottom_w, int bottom_h, int kernel_size);



void checkResults(float* x1, float* x2, int size);

void checkResults(Mat& x1, Mat& x2);
void checkResults(float* x1, Mat& x2);
void checkResultsAlign(float* x1, Mat& x2);

void printMatrix(const char* str, float* a, int m, int n);
