#include "gpu_interface.h"

#define TILE_WIDTH 16
#define MAX_NUM_THREADS 256
#define MASK_WIDTH 7
#define CHANNEL 6
#define MAP_SIZE 16
#define SM_IN (TILE_WIDTH + MASK_WIDTH - 1)
__constant__ float kernelData[MAP_SIZE*CHANNEL*MASK_WIDTH*MASK_WIDTH];

// __constant__ half k_layer1[K_COMMON * K_COMMON * C_LAYER1 * M_LAYER1];
// __constant__ half k_layer2[K_COMMON * K_COMMON * C_LAYER2 * M_LAYER2];

__global__ void conv_forward_kernel4(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    __shared__ float SM_Input [CHANNEL*SM_IN*SM_IN];
    __shared__ float SM_Output [TILE_WIDTH][TILE_WIDTH];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_size = ceil(1.0 * Width_out / TILE_WIDTH);

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int m = blockIdx.x;
    int h_offset = (blockIdx.y / W_size) * TILE_WIDTH;
    int w_offset = (blockIdx.y % W_size) * TILE_WIDTH;
    int b = blockIdx.z;
    //100, 4, 1, 86, 86, 7
    //100, 16, 4, 40, 40, 7

    const int BlockSize = TILE_WIDTH * TILE_WIDTH;
    const int InputSize = Batch * Channel * Height * Width;
    const int msize = Channel * K * K;
    const int SM_Width = TILE_WIDTH + K - 1;
    const int SM_ChSize = SM_Width * SM_Width;
    const int SM_InputSize = Channel * SM_ChSize;

    #define mask_4d(m, c, h, w) kernelData[(m) * (msize) + (c) * (K * K) + (h) * (K) + w]
    #define out_4d(b, m, h, w) output[(b) * (Map_out * Height_out * Width_out) + (m) * (Height_out * Width_out) + (h) * (Width_out) + w]
    #define input_idx(b, c, h, w) ((b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * (Width) + w)
    #define sm_in_idx(c, h, w) ((c)*(SM_ChSize) + (h)*(SM_Width) + w)

    if(tz == 0){
        SM_Output[ty][tx] = 0;
    }
    __syncthreads();

    // shared memory subinput
    for(int newIdx = ty*TILE_WIDTH + tx; newIdx < SM_ChSize; newIdx += BlockSize){
        int newY = newIdx / SM_Width;
        int newX = newIdx % SM_Width;
        for (int c = 0; c < Channel; c++) { // sum over all input channels
            int index = input_idx(b,c,newY+h_offset,newX+w_offset);
            if(index < InputSize)
                SM_Input[sm_in_idx(c,newY,newX)] = input[index];
        }
    }
    __syncthreads();

    int h = h_offset + ty, w = w_offset + tx;
    if(h < Height_out && w < Width_out){
        float acc = 0.0f;
        for (int c = 0; c < Channel; c++) { // sum over all input channels
            for (int p = 0; p < K; p++){ // loop over KxK filter
                for (int q = 0; q < K; q++){
                    acc += SM_Input[sm_in_idx(c, ty+p, tx+q)] * mask_4d(m, c, p, q);
                }
            }
        }
        atomicAdd(&SM_Output[ty][tx], acc);
        __syncthreads();
        if(tz == 0){
            out_4d(b, m, h, w) = SM_Output[ty][tx];
        }
    }

    #undef out_4d
    #undef mask_4d
    #undef input_idx
    #undef sm_in_idx
}


void GPUInterface::conv_forward(
    float *output, const float *input,
                        const float *weight, const int n_sample, const int C_out,
                        const int C_in, const int H_in, const int W_in, const int H_kernel) {

  std::cout << ". Optimize ver 4: \n";

  const int H_out = H_in - H_kernel + 1;
  const int W_out = W_in - H_kernel + 1;

  // Allocate device memory
  float *d_input, *d_output;
  CHECK(cudaMalloc((void **)&d_input, n_sample * C_in * H_in * W_in * sizeof(float)));  // input features map is C_in
  CHECK(cudaMalloc((void **)&d_output, n_sample * C_out * H_out * W_out * sizeof(float)));  // output feature map is C_out

  // Copy input and mask data to device 
  CHECK(cudaMemcpy(d_input, input, n_sample * C_in * H_in * W_in * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(kernelData, weight, C_out * C_in * H_kernel * H_kernel * sizeof(float)));

  // Set the kernel dimensions and call the kernel  
  int X = ceil(1.0 * H_out / TILE_WIDTH);
  int Y = ceil(1.0 * W_out / TILE_WIDTH);
  int Z = X * Y;

  // Block dimensions = #of threads in the block
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);

  // Grid Dimension = #of Blocks: B Size * Num_Output_Features *
  dim3 gridSize(C_out, Z, n_sample);

  // launch the kernel
  GpuTimer timer;
  timer.Start();
  conv_forward_kernel4<<<gridSize, blockSize>>>(
      d_output, d_input, n_sample, C_out, C_in, H_in, W_in, H_kernel);
  timer.Stop();
  float time_kernel_ms = timer.Elapsed();
  std::cout << "\t - Kernel Time: " << time_kernel_ms << " ms" << std::endl;

  // Copy the output back to host
  cudaMemcpy(output, d_output, n_sample * C_out * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  
}