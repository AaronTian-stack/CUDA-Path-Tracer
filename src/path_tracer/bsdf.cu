#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <glm/glm.hpp>

#include "ggx.h"
#include "scene_structs.h"
#include "util.h"

// Sampling and coordinate conversion adapted from my CIS 5610 path tracer

__device__ glm::vec3 square_to_hemisphere_cosine(const glm::vec2& xi)
{
    float r = glm::sqrt(xi.x);
    float theta = 2.0 * PI * xi.y;
    glm::vec2 disk = glm::vec2(r * cos(theta), r * sin(theta));
    return glm::vec3(disk.x, disk.y, glm::max(0.f, glm::sqrt(1.f - dot(disk, disk))));
}
__device__ float square_to_hemisphere_cosine_pdf(const glm::vec3& sample)
{
    return glm::abs(sample.z) / PI;
}

__device__ void coordinate_system(const glm::vec3& v1, glm::vec3* v2, glm::vec3* v3)
{
    if (abs(v1.x) > abs(v1.y))
    {
        *v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    }
    else
    {
        *v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
    *v3 = glm::cross(v1, *v2);
}
__device__ glm::mat3 local_to_world(const glm::vec3& nor)
{
    glm::vec3 tan, bit;
    coordinate_system(nor, &tan, &bit);
    return glm::mat3(tan, bit, nor);
}
__device__ glm::mat3 world_to_local(const glm::vec3& nor)
{
    return transpose(local_to_world(nor));
}

__device__ glm::vec3 sample_f(const Material& material, const IntersectionData& isect, const glm::vec3& woW, const glm::vec3& xi, glm::vec3* wiW, float* pdf)
{
    glm::vec3 N = isect.normal;
    glm::vec3 V = woW; // In rasterizer you would take vector from fragment position

    const glm::vec3 albedo = glm::vec3(material.albedo);
    const float roughness = material.roughness;
    const float metallic = material.metallic;

    float ggx_pdf = 0;
    float diffuse_pdf = 0;
    if (xi.z < metallic)
    {
        if (roughness == 0)
        {
            *wiW = reflect(-woW, N);
            ggx_pdf = 1.f;
            diffuse_pdf = 0.f;
        }
        else
        {
            auto wo = world_to_local(N) * woW;
            if (wo.z == 0) return glm::vec3(0.);

            glm::vec3 wh = sample_wh(wo, glm::vec2(xi), roughness);
            glm::vec3 wi = reflect(-wo, wh);
            *wiW = local_to_world(N) * wi;
            diffuse_pdf = square_to_hemisphere_cosine_pdf(wi);
            if (!same_hemisphere(wo, wi)) return glm::vec3(0.f);
            ggx_pdf = trowbridge_reitz_pdf(wo, wh, roughness) / (4 * dot(wo, wh));
        }
    }
    else
    {
        // Diffuse Sample
        *wiW = square_to_hemisphere_cosine(glm::vec2(xi));
        // TODO: I think ggx_pdf is wrong here but it looks ok I guess
        diffuse_pdf = square_to_hemisphere_cosine_pdf(*wiW);
        *wiW = local_to_world(N) * *wiW;
    }

    *pdf = metallic * ggx_pdf + (1.f - metallic) * diffuse_pdf;

    const auto& L = *wiW;

    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, glm::vec3(metallic));

    const glm::vec3	H = glm::normalize(L + V);
    const float	NdotV = abs(dot(N, V)) + 1e-5f;
    const float NdotL = glm::max(0.f, dot(L, N));
    const float LdotH = glm::max(0.f, dot(L, H));
    const float	NdotH = glm::max(0.f, dot(H, N));

    glm::vec3 F;
    glm::vec3 specular = get_specular(NdotV, NdotL, LdotH, NdotH, roughness, F0, &F);

    // Otherwise dividing by roughness in GGX which results in a NaN
    if (roughness == 0.f)
    {
        float NdotL_spec = glm::max(0.f, dot(N, *wiW));
        if (NdotL_spec > 0.f)
        {
            specular = F / NdotL_spec;
        }
        else
        {
            specular = glm::vec3(0.f);
        }
    }

    float Fd = fr_disney_diffuse(NdotV, NdotL, LdotH, roughness);
    glm::vec3 kD = (glm::vec3(1.f) - F) * (1.f - metallic);
    glm::vec3 diffuse = kD * (albedo * Fd / PI);

    return diffuse + specular;
}

struct ShadeableIntersection;
__global__ void shade(
    int iter,
    int numPaths,
    const ShadeableIntersection* shadeableIntersections,
    const Material* materials,
    PathSegments path_segments)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPaths)
    {
        return;
    }

    // Reconstruct ray and color
    Ray ray = {path_segments.origins[idx], path_segments.directions[idx]};
    glm::vec3 color = path_segments.colors[idx];
    int bounces = path_segments.remaining_bounces[idx];
    auto intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f)
    {
        thrust::default_random_engine rng = make_seeded_random_engine(iter, idx, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        auto xi = glm::vec3(u01(rng), u01(rng), u01(rng));

        // This is basically just one material type: like principled BSDF in Blender
        // Cool stuff like glass needs glTF extensions like transmission

        const auto material = materials[intersection.material_id];
        // Every object has uniform material. But if you hit the part of the object that is not emissive, you would want to continue
        // Also with MIS we would somehow need to get a PDF, which is not possible like this
        if (glm::dot(material.emissive, material.emissive) > 0.f)
        {
            color *= material.emissive;
            bounces = 0;
        }
        else
        {
            IntersectionData isect;
            isect.position = ray.origin + intersection.t * ray.direction;
            isect.normal = intersection.surface_normal;

            glm::vec3 wi;
            float pdf;
            auto brdf = sample_f(material, isect, -ray.direction, xi, &wi, &pdf);
            float NdotL = glm::max(0.f, glm::dot(isect.normal, wi)); // Use wi after sampling

            ray = spawn_ray(isect.position, wi);
            bounces--;

            if (pdf != 0)
            {
                color *= brdf * NdotL / pdf;
            }
        }
    }
    else
    {
        bounces = 0;
    }

    // Write back
    path_segments.origins[idx] = ray.origin;
    path_segments.directions[idx] = ray.direction;
    path_segments.colors[idx] = color;
    path_segments.remaining_bounces[idx] = bounces;
}