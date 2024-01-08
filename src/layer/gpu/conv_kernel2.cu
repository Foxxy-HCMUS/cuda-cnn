#include "gpu_interface.h"

#define M_CONST 16
#define C_CONST 4
#define K_CONST 7
#define TILE_WIDTH 16

__constant__ float kernelData[M_CONST * C_CONST * K_CONST * K_CONST];

__global__ void conv_forward_kernel2(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K)
{
    extern __shared__ float s_data[];
    const int INPUT_TILE_WIDTH = TILE_WIDTH + K -1;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_size = ceil(1.0 * W_out / TILE_WIDTH);

#define y4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernelData[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define smem(i2, i1, i0) s_data[(i2) * (INPUT_TILE_WIDTH * INPUT_TILE_WIDTH) + (i1) * INPUT_TILE_WIDTH + i0]

    int H_grid = ceil(float(H_out) / TILE_WIDTH);
    int W_grid = ceil(float(W_out) / TILE_WIDTH); 
    
    int b = blockIdx.x;                 // B number
    int m = blockIdx.y;                 // output feature

    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix

    int startOfTile_h = (blockIdx.z / W_grid) * TILE_WIDTH; // row of the input image matrix
    int startOfTile_w = (blockIdx.z % W_grid) * TILE_WIDTH; // col of the input image matrix

    #pragma unroll
    for (int c = 0; c < C; c++)
    {
        #pragma unroll
        for(int i = threadIdx.y; i < INPUT_TILE_WIDTH; i += TILE_WIDTH)
        {
            #pragma unroll
            for(int j = threadIdx.x; j < INPUT_TILE_WIDTH; j += TILE_WIDTH)
            {
                if (startOfTile_h + i < H && startOfTile_w + j < W)
                {
                    smem(c, i, j) = x4d(b, c, startOfTile_h + i, startOfTile_w + j);
                }  
            }
        }
    }

    // Make sure all threads loaded data into shared memory
    __syncthreads();

    if (h < H_out && w < W_out) 
    {
        float sum = 0.0f;
        for(int c=0; c < C; c++)             // sum over all input features
        {
            for(int p=0; p < K; p++)         // KxK filter 
                for(int q=0; q < K; q++)
                    sum += smem(c, p+threadIdx.y, q+threadIdx.x) * k4d(m, c, p, q); // 4 dimensions macro resolve thread index
        }
        y4d(b,m,h,w) = sum;
    }

    #undef y4d
    #undef x4d
    #undef k4d
    #undef smem
  
}

void GPUInterface::conv_forward(
    float *output, const float *input,
                        const float *weight, const int n_sample, const int C_out,
                        const int C_in, const int H_in, const int W_in, const int H_kernel) {

  std::cout << ". Optimize ver 2: Constant-shared memory\n";

  const int H_out = H_in - H_kernel + 1;
  const int W_out = W_in - H_kernel + 1;

  // Allocate device memory
  float *d_input, *d_output, *d_weight;
  CHECK(cudaMalloc((void **)&d_input, n_sample * C_in * H_in * W_in * sizeof(float)));  // input features map is C_in
  CHECK(cudaMalloc((void **)&d_output, n_sample * C_out * H_out * W_out * sizeof(float)));  // output feature map is C_out
  CHECK(cudaMalloc((void **)&d_weight, C_out * C_in * H_kernel * H_kernel * sizeof(float)));  // C_in * C_out filter Maps of size H_kernel * H_kernel

  // Copy input and mask data to device 
  CHECK(cudaMemcpy(d_input, input, n_sample * C_in * H_in * W_in * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(kernelData, weight, C_out * C_in * H_kernel * H_kernel * sizeof(float)));

  // Set the kernel dimensions and call the kernel  
  int X = ceil(1.0 * H_out / TILE_WIDTH);
  int Y = ceil(1.0 * W_out / TILE_WIDTH);
  int Z = X * Y;

  // smem_size
  int smem_size = C_in * (TILE_WIDTH + H_kernel - 1) * (TILE_WIDTH + H_kernel - 1) * sizeof(float);

  // Block dimensions = #of threads in the block
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);

  // Grid Dimension = #of Blocks: B Size * Num_Output_Features *
  dim3 gridSize(n_sample, C_out, Z);

  // launch the kernel
  GpuTimer timer;
  timer.Start();
  conv_forward_kernel2<<<gridSize, blockSize, smem_size>>>(
      d_output, d_input, d_weight, n_sample, C_out, C_in, H_in, W_in, H_kernel);
  timer.Stop();
  float time_kernel_ms = timer.Elapsed();
  std::cout << "\t - Kernel Time: " << time_kernel_ms << " ms" << std::endl;

  // Copy the output back to host
  cudaMemcpy(output, d_output, n_sample * C_out * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_weight);
  
}