
#include "gpu_interface.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel1(float *d_output, const float *d_input, const float *d_weight,
                                    const int n_sample, const int channel_out, const int channel_in,
                                    const int height_in, const int width_in, const int height_kernel) 
{
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;
    int height_grid = ceil(1.0*height_out / TILE_WIDTH);
    int width_grid = ceil(1.0*width_out / TILE_WIDTH); 
    
    int b = blockIdx.x;                 // batch number
    int m = blockIdx.y;                 // output feature
    int h = (blockIdx.z / width_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int w = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix
    
    float accum = 0.0f;

    if (h < height_out && w < width_out) 
    {
        // Shared memory for input and weight
        __shared__ float input_shared[TILE_WIDTH + height_kernel - 1][TILE_WIDTH + height_kernel - 1];
        __shared__ float weight_shared[height_kernel][height_kernel];

        // Load input data into shared memory
        int input_row = threadIdx.y;
        int input_col = threadIdx.x;
        int input_channel = threadIdx.z;
        int input_global_row = (blockIdx.z / width_grid) * TILE_WIDTH + input_row;
        int input_global_col = (blockIdx.z % width_grid) * TILE_WIDTH + input_col;
        if (input_global_row < height_in && input_global_col < width_in)
        {
            input_shared[input_row][input_col] = x4d(b, input_channel, input_global_row, input_global_col);
        }
        else
        {
            input_shared[input_row][input_col] = 0.0f;
        }

        // Load weight data into shared memory
        int weight_row = threadIdx.y;
        int weight_col = threadIdx.x;
        if (weight_row < height_kernel && weight_col < height_kernel)
        {
            weight_shared[weight_row][weight_col] = k4d(m, input_channel, weight_row, weight_col);
        }

        __syncthreads();

        // Compute convolution
        for (int c = 0; c < channel_in; c++)             // sum over all input features
        {
            for (int p = 0; p < height_kernel; p++)         // KxK filter 
            {
                for (int q = 0; q < height_kernel; q++)
                {
                    accum += input_shared[input_row + p][input_col + q] * weight_shared[p][q];
                }
            }
        }

        y4d(b, m, h, w) = accum;
    }
}

void GPUInterface::conv_forward_gpu(
    float *output, const float *input,
                        const float *weight, const int n_sample, const int channel_out,
                        const int channel_in, const int height_in, const int width_in, const int height_kernel) {

  std::cout << ". Optimize ver 1:\n";
  // Set the kernel dimensions and call the kernel

  const int height_out = height_in - height_kernel + 1;
  const int width_out = width_in - height_kernel + 1;

  // Allocate device memory
  float *d_input, *d_output, *d_weight;
  cudaMalloc((void **)&d_input, n_sample * channel_in * height_in * width_in * sizeof(float));  // input features map is channel_in
  cudaMalloc((void **)&d_output, n_sample * channel_out * height_out * width_out * sizeof(float));  // output feature map is channel_out
  cudaMalloc((void **)&d_weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float));  // channel_in * channel_out filter Maps of size kernel_height * kernel_height

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
  conv_forward_kernel1<<<gridSize, blockSize>>>(
      d_output, d_input, d_weight, n_sample, channel_out, channel_in, height_in, width_in, height_kernel);
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

