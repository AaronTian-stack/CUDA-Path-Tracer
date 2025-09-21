#include "cuda_test.h"

#include <cstdio>

void kernelWrapper()
{
    helloKernel<<<1, 1>>>();
	cudaDeviceSynchronize();
}

__global__ void helloKernel() 
{
    printf("device\n");
}