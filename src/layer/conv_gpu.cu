
#include "conv_gpu.h"
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void init_kernel(curandState* state, float* weight, float* bias, int weight_size, int bias_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate state for each thread separately
  curand_init(1234, i, 0, &state[i]);

  if (i < weight_size) {
    weight[i] = curand_normal(&state[i]);
  }

  if (i < bias_size) {
    bias[i] = curand_normal(&state[i]);
  }
}

void ConvGPU::init() {
  // Calculate output dimensions
  height_out = (height_in + 2 * pad_h - height_kernel) / stride + 1;
  width_out = (width_in + 2 * pad_w - width_kernel) / stride + 1;
  dim_out = height_out * width_out * channel_out;

  // Initialize weight and bias matrices
  weight.resize(channel_in * height_kernel * width_kernel * channel_out);
  bias.resize(channel_out);

  // Initialize gradients
  grad_weight.resize(channel_in * height_kernel * width_kernel * channel_out);
  grad_bias.resize(channel_out);

  // Initialize other variables
  data_cols.resize(height_out * width_out);
}

__device__ int min_device(int a, int b) {
  return a < b ? a : b;
}

__global__ void im2col_kernel(const float* image, float* data_col, int height_in, int width_in, int channel_in, int height_out, int width_out, int height_kernel, int width_kernel, int pad_h, int pad_w, int stride) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < height_out * width_out * channel_in) {
    const int batch_idx = blockIdx.y;
    const int h_col = index / (width_out * channel_in);
    const int w_col = (index / channel_in) % width_out;
    const int c_col = index % channel_in;

    // Precompute valid image coordinates based on padding and stride
    int h_start = h_col * stride - pad_h;
    int w_start = w_col * stride - pad_w;
    int h_end = min_device(h_start + height_kernel, height_in);
    int w_end = min_device(w_start + width_kernel, width_in);

    for (int h = h_start; h < h_end; ++h) {
      for (int w = w_start; w < w_end; ++w) {
        if (h >= 0 && w >= 0) {
          data_col[index * height_kernel * width_kernel + (h - h_start) * width_kernel + (w - w_start)] = image[batch_idx * height_in * width_in * channel_in + h * width_in * channel_in + w * channel_in];
        } else {
          // Set padding values to 0
          data_col[index * height_kernel * width_kernel + (h - h_start) * width_kernel + (w - w_start)] = 0.0f;
        }
      }
    }
  }
}


void ConvGPU::im2col(const Vector& image, Matrix& data_col) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;


  // Assuming data() returns a const float* in Vector and float* in Matrix
  const float* image_ptr = image.data();
  float* data_col_ptr = data_col.data();

  // Launch the kernel
  im2col_kernel<<<(hw_out * channel_in * hw_kernel + 255) / 256, 256>>>(
      image_ptr, data_col_ptr, height_in, width_in, channel_in, height_out, width_out,
      height_kernel, width_kernel, pad_h, pad_w, stride);
  CHECK(cudaGetLastError());  // Check for errors in kernel launch
}

__global__ void forward_kernel(float* data_col, float* weight, float* bias, float* top, int data_col_size, int weight_size, int bias_size, int top_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < top_size) {
        float sum = 0;
        for (int j = 0; j < data_col_size; j++) {
            sum += data_col[j] * weight[j];
        }
        top[i] = sum + bias[i % bias_size];
    }
}

void ConvGPU::forward(const Matrix& bottom) {
  Matrix data_col;
  im2col(bottom, data_col);

  int data_col_size = data_col.size();
  int weight_size = weight.size();
  int bias_size = bias.size();
  int top_size = output_dim();

  float* d_data_col;
  float* d_weight;
  float* d_bias;
  float* d_top;

  CHECK(cudaMalloc(&d_data_col, data_col_size * sizeof(float)));
  CHECK(cudaMalloc(&d_weight, weight_size * sizeof(float)));
  CHECK(cudaMalloc(&d_bias, bias_size * sizeof(float)));
  CHECK(cudaMalloc(&d_top, top_size * sizeof(float)));

  CHECK(cudaMemcpy(d_data_col, data_col.data(), data_col_size * sizeof(float), cudaMemcpyHostToDevice));
  std::cout << "weight data: " << weight << " = " << weight_size << std::endl;
  CHECK(cudaMemcpy(d_weight, weight.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_bias, bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));

  forward_kernel<<<(top_size + 255) / 256, 256>>>(d_data_col, d_weight, d_bias, d_top, data_col_size, weight_size, bias_size, top_size);

  CHECK(cudaMemcpy(top.data(), d_top, top_size * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(d_data_col));
  CHECK(cudaFree(d_weight));
  CHECK(cudaFree(d_bias));
  CHECK(cudaFree(d_top));
}




std::vector<float> ConvGPU::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  float* d_res;

  CHECK(cudaMalloc(&d_res, res.size() * sizeof(float)));

  CHECK(cudaMemcpy(d_res, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_res + weight.size(), bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(res.data(), d_res, res.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(d_res));

  return res;
}

void ConvGPU::set_parameters(const std::vector<float>& param) {
  if(static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");

  float* d_param;

  CHECK(cudaMalloc(&d_param, param.size() * sizeof(float)));

  CHECK(cudaMemcpy(d_param, param.data(), param.size() * sizeof(float), cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(weight.data(), d_param, weight.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(bias.data(), d_param + weight.size(), bias.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(d_param));
}

std::vector<float> ConvGPU::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  float* d_res;

  CHECK(cudaMalloc(&d_res, res.size() * sizeof(float)));

  CHECK(cudaMemcpy(d_res, grad_weight.data(), grad_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_res + grad_weight.size(), grad_bias.data(), grad_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(res.data(), d_res, res.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(d_res));

  return res;
}

