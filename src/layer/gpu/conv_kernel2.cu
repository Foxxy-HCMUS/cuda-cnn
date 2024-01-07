#include "gpu_interface.h"

#define TILE_WIDTH 20
#define TILE_HEIGHT 20
#define MASK_WIDTH 7
#define MASK_HEIGHT 7
#define CHANNEL 4
#define MAP_SIZE 16
#define SM_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define SM_HEIGHT (TILE_HEIGHT + MASK_HEIGHT - 1)
__constant__ float Const_Mask [MAP_SIZE*CHANNEL*MASK_WIDTH*MASK_HEIGHT];

__global__ void conv_forward_kernel2(float *d_output, const float *d_input, const float *d_weight,
                                    const int n_sample, const int channel_out, const int channel_in,
                                    const int height_in, const int width_in, const int height_kernel, int S) 
{
    __shared__ float SM_Input [CHANNEL*SM_HEIGHT*SM_WIDTH];

    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;
    int height_grid = ceil(1.0*height_out / TILE_HEIGHT);
    int width_grid = ceil(1.0*width_out / TILE_WIDTH);

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.y;
    int h = (blockIdx.z / width_grid) * TILE_HEIGHT + threadIdx.y;
    int w = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.x;

    const int BlockSize = TILE_WIDTH * TILE_HEIGHT;
    const int InputSize = n_sample * channel_in * height_in * width_in;
    const int msize = channel_in * height_kernel * height_kernel;
    const int SM_ChSize = SM_WIDTH * SM_HEIGHT;
    const int SM_InputSize = channel_in * SM_ChSize;

    #define y4d(b, m, h, w) d_output[(b) * (channel_out * height_out * width_out) + (m) * (height_out * width_out) + (h) * (width_out) + w]
    #define x4d(b, c, h, w) ((b) * (channel_in * height_in * width_in) + (c) * (height_in * width_in) + (h) * (width_in) + w)
    #define k4d(m, c, h, w) Const_Mask[(m) * (msize) + (c) * (height_kernel * height_kernel) + (h) * (height_kernel) + w]
    #define sm_in_idx(c, h, w) ((c)*(SM_ChSize) + (h)*(SM_WIDTH) + w)
    
    // Load input data into shared memory using coalesced memory access
    int offset_x = tx + w - MASK_WIDTH / 2;
    int offset_y = ty + h - MASK_HEIGHT / 2;
    int offset = x4d(b, 0, offset_y, offset_x);
    for(int c = 0; c < channel_in; c++){
        if(offset_x >= 0 && offset_x < width_in && offset_y >= 0 && offset_y < height_in){
            SM_Input[sm_in_idx(c, ty, tx)] = d_input[offset];
        }
        else{
            SM_Input[sm_in_idx(c, ty, tx)] = 0.0f;
        }
        offset += height_in * width_in;
    }
    __syncthreads();
    
    // Compute output values using tiled convolution algorithm
    if(h < height_out && w < width_out){
        for(int i = 0; i < height_kernel; i += TILE_HEIGHT){
            for(int j = 0; j < height_kernel; j += TILE_WIDTH){
                float acc = 0.0f;
                for (int c = 0; c < channel_in; c++) { // sum over all input channels
                    for (int p = i; p < i + TILE_HEIGHT; p++){ // loop over KxK filter
                        for (int q = j; q < j + TILE_WIDTH; q++){
                            acc += SM_Input[sm_in_idx(c, ty+p, tx+q)] * k4d(m, c, p, q);
                        }
                    }
                }
                atomicAdd(&y4d(b, m, h, w), acc); // accumulate the partial results
            }
        }
    }

    #undef y4d
    #undef k4d
    #undef x4d
    #undef sm_in_idx
}


void GPUInterface::conv_forward_gpu(
    float *output, const float *input,
                        const float *weight, const int n_sample, const int channel_out,
                        const int channel_in, const int height_in, const int width_in, const int height_kernel) {

  std::cout << ". Optimize ver 2:\n";
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
  dim3 gridSize(channel_out, Z, n_sample);

  // launch the kernel
  GpuTimer timer;
  timer.Start();
  conv_forward_kernel2<<<gridSize, blockSize>>>(
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