#pragma once
#include "scene_structs.h"
#include <cuda_runtime.h>

__host__ __device__ float box_intersection_test(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__host__ __device__ float sphere_intersection_test(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);
