#include "../utils/mat.h"

  void conv3x3s1_winograd64_transform_kernel_neon(const Mat& kernel, Mat& kernel_tm, int inch, int outch);

  void conv3x3s1_winograd64_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias);
