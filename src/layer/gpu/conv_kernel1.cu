
#include "gpu_interface.h"

#define TILE_WIDTH 16


__global__ void conv_forward_kernel1(float *d_output, const float *d_input, const float *d_weight,
                                    const int n_sample, const int channel_out, const int channel_in,
                                    const int height_in, const int width_in, const int height_kernel, int S) 
{
    const int height_out = (height_in - height_kernel)/S + 1;
    const int width_out = (width_in - height_kernel)/S + 1;
#define y4d(i3, i2, i1, i0) d_output[(i3) * (channel_out * height_out * width_out) + (i2) * (height_out * width_out) + (i1) * (width_out) + i0]
#define x4d(i3, i2, i1, i0) d_input[(i3) * (channel_in * height_in * width_in) + (i2) * (height_in * width_in) + (i1) * (width_in) + i0]
#define k4d(i3, i2, i1, i0) d_weight[(i3) * (channel_in * height_kernel * height_kernel) + (i2) * (height_kernel * height_kernel) + (i1) * (height_kernel) + i0]
#define share_2d(i1, i0) shared_mem[(i1) * (TILE_WIDTH * (S + 1)) + i0]
    extern __shared__ float shared_mem[];

    float* W_shared = &shared_mem[(TILE_WIDTH + height_kernel - 1) * (TILE_WIDTH + height_kernel - 1) * (S + 1) * (S + 1)];
    
    int W_grid = ceil(width_out / (TILE_WIDTH * 1.0));
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int n = blockIdx.x, m = blockIdx.y;
    int start_w = TILE_WIDTH * (blockIdx.z % W_grid);
    int start_h = TILE_WIDTH * (blockIdx.z / W_grid);
    int x = threadIdx.x;
    int y = threadIdx.y;
    float acc = 0.0; 
    for(int c = 0; c < channel_in; c++) 
    {
        for(int i = 0; i <= S; i++)
        {
            for(int j = 0; j <= S; j++)
            {
                if(start_h * S + y + j * TILE_WIDTH < height_in && start_w * S + x + i * TILE_WIDTH < width_in)
                    share_2d(j * TILE_WIDTH + y, i * TILE_WIDTH + x) = x4d(n, c, start_h * S + j * TILE_WIDTH + y, start_w * S + i * TILE_WIDTH + x);
                else 
                    share_2d(j * TILE_WIDTH + y, i * TILE_WIDTH + x) = 0.0;
            }
        }
        __syncthreads();

        if(x < height_kernel && y < height_kernel)  
            W_shared[y * height_kernel + x] = k4d(m, c, y, x);
        __syncthreads();

        if(w < width_out && h < height_out)
        {
            for(int p = 0; p < height_kernel; p++) 
            {       
                for(int q = 0; q < height_kernel; q++)
                {
                    acc += share_2d(y * S + p, x * S + q) * W_shared[p * height_kernel + q]; 
                }
            }
        }
        __syncthreads();
        
    }
    
    if(w < width_out && h < height_out)      
        y4d(n, m, h, w) = acc;

    #undef y4d
    #undef x4d
    #undef k4d
}

void GPUInterface::conv_forward(
    float *output, const float *input,
                        const float *weight, const int n_sample, const int channel_out,
                        const int channel_in, const int height_in, const int width_in, const int height_kernel) {

  std::cout << ". Optimize ver 1:\n";
  // Set the kernel dimensions and call the kernel
  int S = 1; // stride

  const int height_out = height_in - height_kernel + 1;
  const int width_out = width_in - height_kernel + 1;

  // Allocate device memory
  float *d_input, *d_output, *d_weight;
  cudaMalloc((void **)&d_input, n_sample * channel_in * height_in * width_in * sizeof(float));  // input features map is channel_in
  cudaMalloc((void **)&d_output, n_sample * channel_out * height_out * width_out * sizeof(float));  // output feature map is channel_out
  cudaMalloc((void **)&d_weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float));  // channel_in * channel_out filter Maps of size height_kernel * height_kernel

  // Copy input and mask data to device 
  cudaMemcpy(d_input, input, n_sample * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float), cudaMemcpyHostToDevice);

  // Set the kernel dimensions and call the kernel  
  int X = ceil(1.0 * height_out / TILE_WIDTH);
  int Y = ceil(1.0 * width_out / TILE_WIDTH);
  int Z = X * Y;

  // Block dimensions = #of threads in the block
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);

  // Grid Dimension = #of Blocks: Batch Size * Num_Output_Features *
  dim3 gridSize(n_sample, channel_out, Z);

  // launch the kernel
  GpuTimer timer;
  timer.Start();
  conv_forward_kernel1<<<gridSize, blockSize, 
  (TILE_WIDTH + height_kernel - 1) * (TILE_WIDTH + height_kernel - 1) * sizeof(float) * (S + 1) * (S + 1) + height_kernel * height_kernel * sizeof(float)>>>(
      d_output, d_input, d_weight, n_sample, channel_out, channel_in, height_in, width_in, height_kernel, S);
  timer.Stop();
  float time_kernel_ms = timer.Elapsed();
  std::cout << "\t - Kernel Time: " << time_kernel_ms << " ms" << std::endl;

  // Copy the output back to host
  cudaMemcpy(output, d_output, n_sample * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_weight);
  
}

