#include "cuda_pt.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/partition.h>

#include "bsdf.h"
#include "camera.h"
#include "intersection.h"
#include "util.h"

// Debug kernel to write cosine gradient to texture
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

void test_set_image(cudaSurfaceObject_t surf_obj, size_t width, size_t height, float time)
{
    dim3 block(16, 16);
    dim3 grid(divup(width, block.x), divup(height, block.y));
    set_image_uv<<<grid, block>>>(surf_obj, width, height, time);
}

__global__ void generate_ray_from_camera(Camera cam, int iter, int traceDepth, PathSegments path_segments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        path_segments.origins[index] = cam.position;
        path_segments.colors[index] = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = make_seeded_random_engine(iter, 0, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // Antialiasing by jittering the ray
        path_segments.directions[index] = glm::normalize(cam.view
            - cam.right * cam.pixel_length.x * (static_cast<float>(x) - static_cast<float>(cam.resolution.x) * 0.5f + u01(rng))
            - cam.up * cam.pixel_length.y * (static_cast<float>(y) - static_cast<float>(cam.resolution.y) * 0.5f + u01(rng))
        );

        path_segments.pixel_indices[index] = index;
        path_segments.remaining_bounces[index] = traceDepth;
    }
}

__global__ void accumulate_albedo_normal(int num_paths, ShadeableIntersection* intersections, Material* materials,
    glm::vec3* albedo_image, glm::vec3* normal_image)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_paths)
    {
        auto& inter = intersections[index];
        if (inter.t > 0.0f)
        {
            auto& mat = materials[inter.material_id];

            albedo_image[index] += glm::vec3(mat.albedo);
            normal_image[index] += inter.surface_normal;
        }
    }
}

__global__ void set_image_from_vec3(cudaSurfaceObject_t surf, glm::vec3* image, size_t width, size_t height, float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    int index = x + y * width;
    glm::vec3 col = image[index] * scale;
    col = glm::clamp(col, 0.0f, 1.0f);
    uchar4 color;
    color.x = col.x * 255.0f;
    color.y = col.y * 255.0f;
    color.z = col.z * 255.0f;
    color.w = 255;
    surf2Dwrite(color, surf, x * sizeof(uchar4), y);
}

void set_image(const dim3& grid, const dim3& block, cudaSurfaceObject_t surf_obj, glm::vec3* image, size_t width, size_t height, float scale)
{
    set_image_from_vec3<<<grid, block>>>(surf_obj, image, width, height, scale);
}

void generate_ray_from_camera(const dim3& grid, const dim3& block, const Camera& cam, int iter, int trace_depth,
                              PathSegments path_segments)
{
	generate_ray_from_camera<<<grid, block>>>(cam, iter, trace_depth, path_segments);
}

void accumulate_albedo_normal(const dim3& grid, const int block_size_1D, int num_paths,
	ShadeableIntersection* intersections, Material* materials, glm::vec3* accumulated_albedo,
	glm::vec3* accumulated_normal)
{
	accumulate_albedo_normal<<<grid, block_size_1D>>>(num_paths, intersections, materials, accumulated_albedo, accumulated_normal);
}

void sort_paths_by_material(ShadeableIntersection* intersections, PathSegments path_segments, int num_paths)
{
    auto keys = intersections;
    auto values = thrust::make_zip_iterator(thrust::make_tuple(path_segments.origins, path_segments.directions, path_segments.colors, path_segments.pixel_indices, path_segments.remaining_bounces));
    thrust::sort_by_key(thrust::device, keys, keys + num_paths, values,
        [] __device__(const ShadeableIntersection& a, const ShadeableIntersection& b) {
            return a.material_id < b.material_id;
        });
}

__global__ void compute_intersections(int depth, int num_paths, PathSegments path_segments, Geom* geoms, int num_geoms, ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        Ray ray = {path_segments.origins[path_index], path_segments.directions[path_index]};

        float t;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < num_geoms; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = box_intersection_test(geom, ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphere_intersection_test(geom, ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].material_id = geoms[hit_geom_index].material_id;
            intersections[path_index].surface_normal = normal;
        }
    }
}

void compute_intersections(int threads, int depth, int num_paths, PathSegments path_segments, Geom* geoms, int num_geoms, ShadeableIntersection* intersections)
{
    dim3 block(threads);
    dim3 grid(divup(num_paths, block.x));
    compute_intersections<<<grid, block>>>(depth, num_paths, path_segments, geoms, num_geoms, intersections);
}

__global__ void final_gather_kernel(int initial_num_paths, glm::vec3* image, PathSegments path_segments)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= initial_num_paths) return;
    image[path_segments.pixel_indices[index]] += path_segments.colors[index];
}

void final_gather(int threads, int initial_num_paths, glm::vec3* image, PathSegments path_segments)
{
    dim3 block(threads);
    dim3 grid(divup(initial_num_paths, block.x));
    final_gather_kernel<<<grid, block>>>(initial_num_paths, image, path_segments);
}

__global__ void normalize_albedo_normal(glm::vec2 resolution, int iter, glm::vec3* accumulated_albedo, glm::vec3* accumulated_normal, glm::vec3* albedo_image, glm::vec3* normal_image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;
    int index = x + y * static_cast<int>(resolution.x);
    albedo_image[index] = accumulated_albedo[index] / static_cast<float>(iter);
    normal_image[index] = accumulated_normal[index] / static_cast<float>(iter);
}

void normalize_albedo_normal(const dim3& grid, const dim3& block, glm::vec2 resolution, int iter, glm::vec3* accumulated_albedo, glm::vec3* accumulated_normal, glm::vec3* albedo_image, glm::vec3* normal_image)
{
    normalize_albedo_normal<<<grid, block>>>(resolution, iter, accumulated_albedo, accumulated_normal, albedo_image, normal_image);
}

__global__ void average_image_for_denoise(glm::vec3* image, glm::vec2 resolution, int iter, glm::vec3* in_denoise)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;
    int index = x + y * static_cast<int>(resolution.x);
    in_denoise[index] = image[index] / static_cast<float>(iter);
}

void average_image_for_denoise(const dim3& grid, const dim3& block, glm::vec3* image, glm::vec2 resolution, int iter, glm::vec3* in_denoise)
{
    average_image_for_denoise<<<grid, block>>>(image, resolution, iter, in_denoise);
}

void shade_paths(int threads, int iteration, int num_paths, ShadeableIntersection* intersections, Material* materials, PathSegments path_segments)
{
    dim3 block(128);
    dim3 grid(divup(num_paths, block.x));
    shade<<<grid, block>>>(iteration, num_paths, intersections, materials, path_segments);
}

int filter_paths_with_bounces(PathSegments path_segments, int num_paths)
{
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(path_segments.origins, path_segments.directions, path_segments.colors, path_segments.pixel_indices, path_segments.remaining_bounces));
    auto zip_end = zip_begin + num_paths;
    auto new_end = thrust::partition(thrust::device, zip_begin, zip_end,
        [] __device__(const thrust::tuple<glm::vec3, glm::vec3, glm::vec3, int, int>& t) {
            return thrust::get<4>(t) > 0;
        });
    return static_cast<int>(new_end - zip_begin);
}
