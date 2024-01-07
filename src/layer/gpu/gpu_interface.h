#ifndef SRC_LAYER_GPU_INTERFACE_H_
#define SRC_LAYER_GPU_INTERFACE_H_

#include <vector>
#include <math.h>
#include <iostream>
#include "./utils.h"

class GPUInterface {
 public:
  
  void conv_forward_gpu(float *d_output, const float *d_input,
                        const float *d_weight, const int n_sample,
                        const int channel_out, const int channel_in,
                        const int height_in, const int width_in,
                        const int height_kernel);
};

#endif  // SRC_LAYER_GPU_INTERFACE_H_
