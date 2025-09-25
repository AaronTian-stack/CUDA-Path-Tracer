#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ bool same_hemisphere(glm::vec3 w, glm::vec3 wp);

__device__ float trowbridge_reitz_pdf(glm::vec3 wo, glm::vec3 wh, float roughness);
__device__ glm::vec3 sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness);
__device__ glm::vec3 f_microfacet_refl(glm::vec3 albedo, glm::vec3 wo, glm::vec3 wi, float roughness);

__device__ float fr_disney_diffuse(float NdotV, float NdotL, float LdotH, float linear_roughness);
__device__ glm::vec3 get_specular(float NdotV, float NdotL, float LdotH, float NdotH, float roughness, glm::vec3 f0, glm::vec3* F);
