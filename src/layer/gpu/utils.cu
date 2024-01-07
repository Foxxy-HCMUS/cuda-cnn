#include "utils.h"

char *GPUUtils::concatStr(const char *s1, const char *s2)
{
	char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

void GPUUtils::printDeviceInfo()
{
	cudaDeviceProp devProv;
	CHECK(cudaGetDeviceProperties(&devProv, 0));
	printf("**********GPU info**********\n");
	printf("Name: %s\n", devProv.name);
	printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
	printf("Num SMs: %d\n", devProv.multiProcessorCount);
	printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
	printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
	printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
	printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
	printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
	printf("****************************\n");
}

__global__ void do_not_remove_this_kernel() {
    int tx = threadIdx.x;
    tx = tx + 1;
}

__global__ void prefn_marker_kernel() {
    int tx = threadIdx.x;
    tx = tx + 1;
}

__host__ void GPUUtils::insert_post_barrier_kernel() {
    
    dim3 GridDim(1,1,1);
    dim3 BlockDim(1,1,1);
    do_not_remove_this_kernel<<<GridDim, BlockDim>>>();
    cudaDeviceSynchronize();
}

__host__ void GPUUtils::insert_pre_barrier_kernel() {

    int* devicePtr;
    int x = 1;

    cudaMalloc((void**) &devicePtr, sizeof(int));
    cudaMemcpy(devicePtr, &x, sizeof(int), cudaMemcpyHostToDevice);

    dim3 GridDim(1,1,1);
    dim3 BlockDim(1,1,1);
    prefn_marker_kernel<<<GridDim, BlockDim>>>();
    cudaFree(devicePtr);
    cudaDeviceSynchronize();
}