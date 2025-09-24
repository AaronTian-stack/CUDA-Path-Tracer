#include "cuda_pt.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__global__ void set_image_uv(cudaSurfaceObject_t surf, size_t width, size_t height, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    // From default ShaderToy shader
	glm::vec2 uv = glm::vec2(x / static_cast<float>(width), y / static_cast<float>(height));
    glm::vec3 col = 0.5f + 0.5f * cos(time + glm::vec3(glm::vec2(uv), uv.x) + glm::vec3(0, 2, 4));
    uchar4 color;
    color.x = col.x * 255.0f;
    color.y = col.y * 255.0f;
    color.z = col.z * 255.0f;
    color.w = 255;
    surf2Dwrite(color, surf, x * sizeof(uchar4), y);
}

inline int divup(const int a, const int b) 
{
    return (a + b - 1) / b;
}

void test_set_image(cudaSurfaceObject_t surf_obj, size_t width, size_t height, float time, cudaExternalSemaphore_t sem)
{
    dim3 block(16, 16);
    dim3 grid(divup(width, block.x), divup(height, block.y));
    set_image_uv<<<grid, block>>>(surf_obj, width, height, time);
    cudaExternalSemaphoreSignalParams params{};
    cudaSignalExternalSemaphoresAsync(&sem, &params, 1, 0);
}