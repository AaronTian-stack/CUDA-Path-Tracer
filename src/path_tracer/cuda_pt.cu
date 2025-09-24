#include "cuda_pt.h"
#include <cstdio>

#include <cuda_runtime.h>

__global__ void set_image_white_kernel(cudaSurfaceObject_t surf, size_t width, size_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    uchar4 color;
    color.x = static_cast<unsigned char>(x / static_cast<float>(width) * 255.0f);
    color.y = static_cast<unsigned char>(y / static_cast<float>(height) * 255.0f);
    color.z = 0;
    color.w = 255;
    surf2Dwrite(color, surf, x * sizeof(uchar4), y);
}

inline int divup(const int a, const int b) 
{
    return (a + b - 1) / b;
}

void test_set_image_white(cudaSurfaceObject_t surf_obj, size_t width, size_t height)
{
    dim3 block(16, 16);
    dim3 grid(divup(width, block.x), divup(height, block.y));
    set_image_white_kernel<<<grid, block>>>(surf_obj, width, height);
    cudaDeviceSynchronize();
}