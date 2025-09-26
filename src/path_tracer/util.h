#pragma once

#include <thrust/random.h>
#include <cuda_runtime.h>

#include "scene_structs.h"

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f

__host__ __device__ inline int divup(const int a, const int b)
{
    return (a + b - 1) / b;
}

__host__ __device__ inline unsigned int hash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ inline thrust::default_random_engine make_seeded_random_engine(int iter, int index, int depth)
{
    int h = hash((1 << 31) | (depth << 22) | iter) ^ hash(index);
    return thrust::default_random_engine(h);
}

__host__ __device__ inline glm::vec3 get_point_on_ray(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

__host__ __device__ inline Ray spawn_ray(const glm::vec3& pos, const glm::vec3& wi)
{
    return { pos + wi * 0.0001f, wi };
}
