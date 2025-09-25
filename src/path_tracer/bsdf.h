#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "scene_structs.h"

__global__ void shade(int iter, int numPaths, const ShadeableIntersection* shadeableIntersections, const Material* materials, PathSegments path_segments);

__device__ glm::vec3 sample_f(const Material& material, const IntersectionData& isect, const glm::vec3& woW, const glm::vec3& xi, glm::vec3* wiW, float* pdf);
