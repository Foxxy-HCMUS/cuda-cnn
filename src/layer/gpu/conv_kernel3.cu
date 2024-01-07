#include <cmath>
#include <iostream>
#include <stdio.h>
#include <pthread.h>
#include "gpu_interface.h"
#include "cuda_fp16.h"

#define TILE_WIDTH 16
#define TILE_WIDTH_2 8
#define K_COMMON 7
#define C_LAYER1 1
#define M_LAYER1 6
#define C_LAYER2 6
#define M_LAYER2 16
#define NUM_PTHREADS 32

__constant__ half k_layer1[K_COMMON * K_COMMON * C_LAYER1 * M_LAYER1];
__constant__ half k_layer2[K_COMMON * K_COMMON * C_LAYER2 * M_LAYER2];

__global__ void conv_forward_kernel_layer1(float *y, const float * __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k_layer1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define loc_mem(i2, i1, i0) loc_mem1[(i2) * ((TILE_WIDTH+K_COMMON-1) * (TILE_WIDTH+K_COMMON-1)) + (i1) * (TILE_WIDTH+K_COMMON-1) + (i0)]

    // Insert your GPU convolution kernel code here
    int b,m,h,w;
    b = blockIdx.x;
    m = blockIdx.y;
    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    h = (blockIdx.z/W_grid)*TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

    extern __shared__ half loc_mem1[(TILE_WIDTH + K_COMMON - 1) * (TILE_WIDTH + K_COMMON - 1)];

    loc_mem(0, threadIdx.y, threadIdx.x) = __float2half(x4d(b, 0, h, w));
    loc_mem(0, threadIdx.y + (K-1), threadIdx.x) = __float2half(x4d(b, 0, h + (K - 1), w));
    loc_mem(0, threadIdx.y, threadIdx.x + (K - 1)) = __float2half(x4d(b, 0, h, w + (K - 1)));
    loc_mem(0, threadIdx.y + (K - 1), threadIdx.x + (K - 1)) = __float2half(x4d(b, 0, h + (K - 1), w + (K - 1)));

    __syncthreads();

    if (h < H_out && w < W_out){
      half acc = 0.0f;
      #pragma unroll 7
      for (int p = 0; p < K; p++) {
          #pragma unroll 7
          for (int q = 0; q < K; q++) {
              acc = __hadd(acc, __hmul(loc_mem(0, threadIdx.y+p, threadIdx.x+q), k4d(m, 0, p, q))); // C = C_LAYER1 - 1
          }
      }
      y4d(b, m, h, w) = __half2float(acc);
    }

#undef y4d
#undef x4d
#undef k4d
#undef loc_mem
}

__global__ void conv_forward_kernel_layer2(float *y, const float * __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k_layer2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define loc_mem(i2, i1, i0) loc_mem2[(i2) * ((TILE_WIDTH_2+K-1) * (TILE_WIDTH_2+K-1)) + (i1) * (TILE_WIDTH_2+K-1) + (i0)]

    // Insert your GPU convolution kernel code here

    int b,m,h,w;
    b = blockIdx.x;
    m = blockIdx.y;
    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    h = (blockIdx.z/W_grid)*TILE_WIDTH_2 + threadIdx.y;
    w = (blockIdx.z % W_grid)*TILE_WIDTH_2 + threadIdx.x;

    extern __shared__ half loc_mem2[C_LAYER2 * (TILE_WIDTH_2 + K_COMMON - 1) * (TILE_WIDTH_2 + K_COMMON - 1)]; // set size on launch

    #pragma unroll 6 //6 //7
    for(int c = 0; c < C; c++) {
      loc_mem(c, threadIdx.y, threadIdx.x) = __float2half(x4d(b, c, h, w));
      loc_mem(c, threadIdx.y + (K-1), threadIdx.x) = __float2half(x4d(b, c, h + (K - 1), w));
      loc_mem(c, threadIdx.y, threadIdx.x + (K - 1)) = __float2half(x4d(b, c, h, w + (K - 1)));
      loc_mem(c, threadIdx.y + (K - 1), threadIdx.x + (K - 1)) = __float2half(x4d(b, c, h + (K - 1), w + (K - 1)));
    }

    __syncthreads();

    if (h < H_out && w < W_out) {
        half acc = 0.0f;
        #pragma unroll 6 //6
        for (int c = 0; c < C; c++) {
            #pragma unroll 7
            for (int p = 0; p < K; p++) {
                #pragma unroll 7
                for (int q = 0; q < K; q++) {
                    acc = __hadd(acc, __hmul(loc_mem(c, threadIdx.y+p, threadIdx.x+q), k4d(m, c, p, q)));
                }
            }
        }
        y4d(b, m, h, w) = __half2float(acc);
    }

#undef y4d
#undef x4d
#undef k4d
#undef loc_mem
}

struct args {
  const float *f_arr_k;
  half *h_arr_k;
  int len_k;
  int start_idx_k;
};

struct args2 {
  float *f_arr;
  half * h_arr;
  int len;
  int start_idx;
};

struct args_pinned {
  const float *copy_from_address;
  float *copy_to_address;
  int len;
  int start_idx;
  int end_idx;
};

__host__ void *pinned_transfer(void *args_pinned_list) {
  int len = ((struct args_pinned *)args_pinned_list)->len;
  int start_idx = ((struct args_pinned *)args_pinned_list)->start_idx;
  int end_idx = ((struct args_pinned *)args_pinned_list)->end_idx;
  float *copy_to_address = ((struct args_pinned *)args_pinned_list)->copy_to_address;
  const float *copy_from_address = ((struct args_pinned *)args_pinned_list)->copy_from_address;

  #pragma unroll 7
  for (int i = start_idx; i < end_idx; i++) {
    if (i >= len) 
      break;
    copy_to_address[i] = copy_from_address[i];
  }
  return NULL;
}

__host__ void * f2h_host(void *arg_list){

  int len_k = ((struct args *)arg_list)->len_k;
  #pragma unroll 7
  for(int i = 0; i < ceil((1.0*len_k)/(NUM_PTHREADS)); i++){
    int idx  = ((struct args *)arg_list)->start_idx_k + i;
    if(idx >= len_k)
      break;
    (((struct args *)arg_list)->h_arr_k)[idx] = __float2half((((struct args *)arg_list)->f_arr_k)[idx]);
  }
  return NULL;
}

__host__ void * h2f_host(void *arg_list){ // bypasses const by converting from void *
  int len = ((struct args2 *)arg_list)->len;
  #pragma unroll 7
  for(int i = 0; i < ceil((1.0*len)/(NUM_PTHREADS)); i++){
    int idx  = ((struct args2 *)arg_list)->start_idx + i;
    if(idx >= len)
      break;
    (((struct args2 *)arg_list)->f_arr)[idx] = __float2half((((struct args2 *)arg_list)->h_arr)[idx]);
  }
  return NULL;
}

void GPUInterface::conv_forward(
    float *output, const float *input,
                        const float *weight, const int n_sample, const int channel_out,
                        const int channel_in, const int height_in, const int width_in, const int height_kernel) {

  std::cout << ". Optimize ver 3:\n";
  // Set the kernel dimensions and call the kernel

  const int height_out = height_in - height_kernel + 1;
  const int width_out = width_in - height_kernel + 1;

  // Allocate device memory
  float *d_input, *d_output;
  // Allocate pinned host memory 
  float *host_pinned_x;
  float *host_pinned_y;

  int x_len = width_in * height_in * channel_in * n_sample;
  int y_len = n_sample * channel_out * height_out * width_out;
  int k_len = channel_out * channel_in * height_kernel * height_kernel;

  half *host_k16 = (half *)malloc(k_len * sizeof(half));

  pthread_t tids[NUM_PTHREADS];
  struct args arg_list[NUM_PTHREADS];

  #pragma unroll NUM_PTHREADS
  for (unsigned int i = 0; i <  NUM_PTHREADS; i++) {
    arg_list[i].len_k = k_len;
    arg_list[i].f_arr_k = weight;
    arg_list[i].h_arr_k = host_k16;
    arg_list[i].start_idx_k = i*ceil((1.0*k_len)/NUM_PTHREADS);
    pthread_create(tids + i, NULL, f2h_host, (void *)(arg_list + i));
  }

  #pragma unroll NUM_PTHREADS
  for (unsigned int i = 0; i <  NUM_PTHREADS; i++) {
    pthread_join(tids[i], NULL);
  }
  
  cudaMalloc((void **)&d_input, x_len * sizeof(float));  // input features map is channel_in
  cudaMalloc((void **)&d_output, y_len * sizeof(float));  // output feature map is channel_out

  int tile_width = TILE_WIDTH;
  if (channel_in != 1) tile_width = TILE_WIDTH_2;

  // Set the kernel dimensions and call the kernel  
  int X = ceil(1.0 * height_out / tile_width);
  int Y = ceil(1.0 * width_out / tile_width);
  int Z = X * Y;

  // Block dimensions = #of threads in the block
  dim3 blockSize(tile_width, tile_width, 1);

  // Grid Dimension = #of Blocks: Batch Size * Num_Output_Features *
  dim3 gridSize(n_sample, channel_out, Z);

  //Use Pinned Memory
  cudaHostAlloc((void **)&host_pinned_x, x_len * sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_pinned_y, y_len * sizeof(float), cudaHostAllocDefault);

  struct args_pinned args_pinned_list[NUM_PTHREADS];

  #pragma unroll NUM_PTHREADS
  for (unsigned int i = 0; i <  NUM_PTHREADS; i++) {
    args_pinned_list[i].len = x_len;
    args_pinned_list[i].copy_from_address = input;
    args_pinned_list[i].copy_to_address = host_pinned_x;
    args_pinned_list[i].start_idx = i*ceil((1.0*x_len)/NUM_PTHREADS);
    args_pinned_list[i].end_idx = (i+1)*ceil((1.0*x_len)/NUM_PTHREADS);
    pthread_create(tids + i, NULL, pinned_transfer, (void *)(args_pinned_list + i));
  }

  #pragma unroll NUM_PTHREADS
  for (unsigned int i = 0; i <  NUM_PTHREADS; i++) {
    pthread_join(tids[i], NULL);
  }

  // put d_weight into constant memory
  if (channel_in == 1) { cudaMemcpyToSymbol(k_layer1, host_k16, height_kernel * height_kernel * C_LAYER1 * M_LAYER1 * sizeof(half)); }
  else { cudaMemcpyToSymbol(k_layer2, host_k16, height_kernel * height_kernel * C_LAYER2 * M_LAYER2 * sizeof(half)); }

  cudaMemcpy(d_input, host_pinned_x, x_len * sizeof(float), cudaMemcpyHostToDevice);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // launch the kernel
  GpuTimer timer;
  timer.Start();

  if (channel_in == 1) { conv_forward_kernel_layer1<<<gridSize, blockSize>>>(
      d_output, d_input, n_sample, channel_out, channel_in, height_in, width_in, height_kernel); }
  else { conv_forward_kernel_layer2<<<gridSize, blockSize>>>(
      d_output, d_input, n_sample, channel_out, channel_in, height_in, width_in, height_kernel); }
  timer.Stop();

  float time_kernel_ms = timer.Elapsed();
  std::cout << "\t - Kernel Time: " << time_kernel_ms << " ms" << std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  // Copy the output back to host
  cudaMemcpy(host_pinned_y, d_output, y_len * sizeof(float), cudaMemcpyDeviceToHost);

  #pragma unroll NUM_PTHREADS
  for (unsigned int i = 0; i <  NUM_PTHREADS; i++) {
    args_pinned_list[i].len = y_len;
    args_pinned_list[i].copy_from_address = host_pinned_y;
    args_pinned_list[i].copy_to_address = output;
    args_pinned_list[i].start_idx = i*ceil((1.0*y_len)/NUM_PTHREADS);
    args_pinned_list[i].end_idx = (i+1)*ceil((1.0*y_len)/NUM_PTHREADS);
    pthread_create(tids + i, NULL, pinned_transfer, (void *)(args_pinned_list + i));
  }

  #pragma unroll NUM_PTHREADS
  for (unsigned int i = 0; i <  NUM_PTHREADS; i++) {
    pthread_join(tids[i], NULL);
  }

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFreeHost(host_pinned_x);
  cudaFreeHost(host_pinned_y);
  
  free(host_k16);
}


