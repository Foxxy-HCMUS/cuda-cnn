
#include "gpu_interface.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *d_output, const float *d_input, const float *d_weight,
                                    const int n_sample, const int channel_out, const int channel_in,
                                    const int height_in, const int width_in, const int height_kernel)
{
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) d_output[(i3) * (channel_out * height_out * width_out) + (i2) * (height_out * width_out) + (i1) * (width_out) + i0]
#define x4d(i3, i2, i1, i0) d_input[(i3) * (channel_in * height_in * width_in) + (i2) * (height_in * width_in) + (i1) * (width_in) + i0]
#define k4d(i3, i2, i1, i0) d_weight[(i3) * (channel_in * height_kernel * height_kernel) + (i2) * (height_kernel * height_kernel) + (i1) * (height_kernel) + i0]

    int height_grid = ceil(1.0*height_out / TILE_WIDTH);
    int width_grid = ceil(1.0*width_out / TILE_WIDTH); 
    
    int b = blockIdx.x;                 // batch number
    int m = blockIdx.y;                 // output feature
    int h = (blockIdx.z / width_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int w = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix
    
    float accum = 0.0f;

    if (h < height_out && w < width_out) 
    {
        for(int c=0; c<channel_in; c++)             // sum over all input features
        {
            for(int p=0; p<height_kernel; p++)         // KxK filter 
                for(int q=0; q<height_kernel; q++)
                    accum += x4d(b, c, h+p, w+q) * k4d(m, c, p, q); // 4 dimensions macro resolve thread index
        }
        y4d(b,m,h,w) = accum;
    } // endif (h < H_out && w < W_out)

    #undef y4d
    #undef x4d
    #undef k4d
}

void GPUInterface::conv_forward(
    float *output, const float *input,
                        const float *weight, const int n_sample, const int channel_out,
                        const int channel_in, const int height_in, const int width_in, const int height_kernel) {

  std::cout << ". Not Optimize:\n";
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
  conv_forward_kernel<<<gridSize, blockSize>>>(
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

