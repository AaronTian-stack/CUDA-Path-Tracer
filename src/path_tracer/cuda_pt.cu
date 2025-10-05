#include "cuda_pt.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/partition.h>

#include "bsdf.h"
#include "camera.h"
#include "intersection.h"
#include "util.h"

#define ENABLE_AABB_CULLING 1

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

// From CIS 5610 / Pbrt
__device__ inline glm::vec3 random_in_unit_disk(glm::vec2 sample)
{
    float r = sqrt(sample.x);
    float theta = 2.0f * PI * sample.y;
    return glm::vec3(r * cos(theta), r * sin(theta), 0);
}

__global__ void generate_ray_from_camera(Camera cam, int iter, int traceDepth, PathSegments path_segments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y)
    {
        int index = x + (y * cam.resolution.x);

        thrust::default_random_engine rng = make_seeded_random_engine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // Antialiasing by jittering the ray
        glm::vec3 pixel_dir = glm::normalize(cam.view
            - cam.right * cam.pixel_length.x * (static_cast<float>(x) - static_cast<float>(cam.resolution.x) * 0.5f + u01(rng))
            - cam.up * cam.pixel_length.y * (static_cast<float>(y) - static_cast<float>(cam.resolution.y) * 0.5f + u01(rng))
        );

        glm::vec3 origin = cam.position;
        glm::vec3 direction = pixel_dir;

        {
			float defocus_angle = glm::max(0.0f, cam.defocus_angle);
			// Depth of field thin lens model
			// https://raytracing.github.io/books/RayTracingInOneWeekend.html#defocusblur
            float defocus_radius = cam.focus_distance * tan(glm::radians(defocus_angle / 2.0f));
            glm::vec3 defocus_disk_u = cam.right * defocus_radius;
            glm::vec3 defocus_disk_v = cam.up * defocus_radius;

            glm::vec2 rd_sample = glm::vec2(u01(rng), u01(rng));
            glm::vec3 rd = random_in_unit_disk(rd_sample);
            glm::vec3 offset = rd.x * defocus_disk_u + rd.y * defocus_disk_v;

            origin = cam.position + offset;
            glm::vec3 point_on_focal_plane = cam.position + cam.focus_distance * pixel_dir;
            direction = glm::normalize(point_on_focal_plane - origin);
        }

        path_segments.origins[index] = origin;
        path_segments.colors[index] = glm::vec3(1.0f, 1.0f, 1.0f);
        path_segments.directions[index] = direction;
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

            albedo_image[index] += glm::vec3(mat.base_color.factor);
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

__global__ void compute_intersections(int num_paths, PathSegments path_segments, Geom* geoms, int num_geoms, ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
#ifdef DISABLE_STREAM_COMPACTION
        if (path_segments.remaining_bounces[path_index] <= 0)
        {
            return;
        }
#endif

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
            intersections[path_index].uv = glm::vec2(0.0f);
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].material_id = geoms[hit_geom_index].material_id;
            intersections[path_index].surface_normal = normal;
            intersections[path_index].uv = glm::vec2(0.0f);
        }
    }
}

void compute_intersections(int threads, int depth, int num_paths, PathSegments path_segments, Geom* geoms, int num_geoms, ShadeableIntersection* intersections)
{
    dim3 block(threads);
    dim3 grid(divup(num_paths, block.x));
    compute_intersections<<<grid, block>>>(num_paths, path_segments, geoms, num_geoms, intersections);
}

__device__ float ray_aabb_intersect(const Ray& ray, const glm::vec3& min, const glm::vec3& max)
{
    float t_min = -FLT_MAX;
	float t_max = FLT_MAX;

	#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        if (glm::abs(ray.direction[i]) < 1e-8f) 
        {
            // Ray parallel to axis
            if (ray.origin[i] < min[i] || ray.origin[i] > max[i]) 
            {
                return -1.0f;
            }
        }
    	else 
        {
    		float inv_d = 1.0f / ray.direction[i];
            float t0 = (min[i] - ray.origin[i]) * inv_d;
            float t1 = (max[i] - ray.origin[i]) * inv_d;
            if (inv_d < 0.0f) 
            {
                thrust::swap(t0, t1);
            }
            t_min = glm::max(t_min, t0);
            t_max = glm::min(t_max, t1);
            if (t_max <= t_min) 
            {
                return -1.0f;
            }
        }
    }
    return t_min > 0.0f || t_max > 0.0f ? glm::max(t_min, 0.0f) : -1.0f;
}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ float triangle_intersect(const Ray& ray, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, glm::vec2& bary_out)
{
    const float EPSILON = 1e-6f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
    {
        return -1.0f; // Ray parallel to triangle
    }
    float f = 1.0f / a;
    glm::vec3 s = ray.origin - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f)
    {
	    return -1.0f;
	}
    float t = f * glm::dot(edge2, q);
    if (t > EPSILON)
    {
        bary_out = glm::vec2(u, v);
        return t;
    }
    return -1.0f;
}

__global__ void compute_gltf_intersections_kernel(int num_paths, PathSegments path_segments, pt::glTFModel::DeviceNode* d_nodes, int num_nodes, pt::glTFModel::Primitive* d_primitives, pt::glTFModel::Accessor* d_accessors, pt::glTFModel::BufferView* d_buffer_views, void** d_cu_buffers, ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

#ifdef DISABLE_STREAM_COMPACTION
    if (path_segments.remaining_bounces[path_index] <= 0)
    {
        return;
    }
#endif

    Ray ray = {path_segments.origins[path_index], path_segments.directions[path_index]};
    ShadeableIntersection inter;
    inter.t = -1.0f;
    float closest_t = FLT_MAX;

    // Need to hard code this because I can't import tiny_gltf here
    const int TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT = 5123;

    for (int node_idx = 0; node_idx < num_nodes; node_idx++)
    {
        pt::glTFModel::DeviceNode node = d_nodes[node_idx];
        if (node.mesh_index < 0)
        {
            continue;
        }

        pt::glTFModel::Primitive prim = d_primitives[node.mesh_index];

        glm::mat4 inv_transform = glm::inverse(node.global_transform);
        glm::vec3 local_origin = glm::vec3(inv_transform * glm::vec4(ray.origin, 1.0f));
        glm::vec3 local_dir = glm::vec3(inv_transform * glm::vec4(ray.direction, 0.0f));
        Ray local_ray = {local_origin, local_dir};

        // Indices
        pt::glTFModel::Accessor idx_acc = d_accessors[prim.indices];
        pt::glTFModel::BufferView idx_bv = d_buffer_views[idx_acc.buffer_view];
        void* idx_buffer = d_cu_buffers[idx_bv.buffer_index];
        size_t idx_offset = idx_acc.offset + idx_bv.offset;
        size_t idx_stride = idx_bv.stride;
        if (idx_stride == 0)
        {
            idx_stride = idx_acc.component_type == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT ? 2 : 4;
        }
        int num_triangles = idx_acc.count / 3;

        int pos_accessor_idx = prim.position_accessor;
        int norm_accessor_idx = prim.normal_accessor;
        int texcoord_accessor_idx = prim.texcoord_accessor;
        int tangent_accessor_idx = prim.tangent_accessor;

        if (pos_accessor_idx == -1)
        {
            continue;
        }

        // Positions
        pt::glTFModel::Accessor pos_acc = d_accessors[pos_accessor_idx];
        pt::glTFModel::BufferView pos_bv = d_buffer_views[pos_acc.buffer_view];
        void* pos_buffer = d_cu_buffers[pos_bv.buffer_index];
        size_t pos_offset = pos_acc.offset + pos_bv.offset;
        size_t pos_stride = pos_bv.stride;
        if (pos_stride == 0) 
        {
            pos_stride = 3 * sizeof(float);
        }

        // Normals
        pt::glTFModel::Accessor norm_acc;
        pt::glTFModel::BufferView norm_bv;
        void* norm_buffer = nullptr;
        size_t norm_offset = 0;
        size_t norm_stride = 0;
        if (norm_accessor_idx != -1) 
        {
            norm_acc = d_accessors[norm_accessor_idx];
            norm_bv = d_buffer_views[norm_acc.buffer_view];
            norm_buffer = d_cu_buffers[norm_bv.buffer_index];
            norm_offset = norm_acc.offset + norm_bv.offset;
            norm_stride = norm_bv.stride ? norm_bv.stride : 3 * sizeof(float);
        }

        // UVs
        pt::glTFModel::Accessor texcoord_acc;
        pt::glTFModel::BufferView texcoord_bv;
        void* texcoord_buffer = nullptr;
        size_t texcoord_offset = 0;
        size_t texcoord_stride = 0;
        if (texcoord_accessor_idx != -1) 
        {
            texcoord_acc = d_accessors[texcoord_accessor_idx];
            texcoord_bv = d_buffer_views[texcoord_acc.buffer_view];
            texcoord_buffer = d_cu_buffers[texcoord_bv.buffer_index];
            texcoord_offset = texcoord_acc.offset + texcoord_bv.offset;
            texcoord_stride = texcoord_bv.stride ? texcoord_bv.stride : 2 * sizeof(float);
        }

        // Tangents
        pt::glTFModel::Accessor tangent_acc;
        pt::glTFModel::BufferView tangent_bv;
        void* tangent_buffer = nullptr;
        size_t tangent_offset = 0;
        size_t tangent_stride = 0;
        if (tangent_accessor_idx != -1) 
        {
            tangent_acc = d_accessors[tangent_accessor_idx];
            tangent_bv = d_buffer_views[tangent_acc.buffer_view];
            tangent_buffer = d_cu_buffers[tangent_bv.buffer_index];
            tangent_offset = tangent_acc.offset + tangent_bv.offset;
            tangent_stride = tangent_bv.stride ? tangent_bv.stride : 4 * sizeof(float);
        }

#if ENABLE_AABB_CULLING
        if (ray_aabb_intersect(local_ray, prim.aabb.min, prim.aabb.max) < 0.0f)
        {
            continue;
        }
#endif

        for (int i = 0; i < num_triangles; i++)
        {
            // Get indices
            int i0, i1, i2;
            if (idx_acc.component_type == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT)
            {
                auto indices = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(idx_buffer) + idx_offset + i * 3 * idx_stride);
                i0 = indices[0];
                i1 = indices[1];
                i2 = indices[2];
            }
            else
            {
                auto indices = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(idx_buffer) + idx_offset + i * 3 * idx_stride);
                i0 = indices[0];
                i1 = indices[1];
                i2 = indices[2];
            }

            // Positions
            glm::vec3 v0 = *reinterpret_cast<glm::vec3*>(static_cast<uint8_t*>(pos_buffer) + pos_offset + i0 * pos_stride);
            glm::vec3 v1 = *reinterpret_cast<glm::vec3*>(static_cast<uint8_t*>(pos_buffer) + pos_offset + i1 * pos_stride);
            glm::vec3 v2 = *reinterpret_cast<glm::vec3*>(static_cast<uint8_t*>(pos_buffer) + pos_offset + i2 * pos_stride);

            glm::vec3 geometric_normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

			glm::vec3 n0, n1, n2;
            n0 = n1 = n2 = geometric_normal;

            if (norm_accessor_idx != -1) 
            {
                n0 = *reinterpret_cast<glm::vec3*>(static_cast<uint8_t*>(norm_buffer) + norm_offset + i0 * norm_stride);
                n1 = *reinterpret_cast<glm::vec3*>(static_cast<uint8_t*>(norm_buffer) + norm_offset + i1 * norm_stride);
                n2 = *reinterpret_cast<glm::vec3*>(static_cast<uint8_t*>(norm_buffer) + norm_offset + i2 * norm_stride);
            }

            glm::vec2 uv0, uv1, uv2;
            uv0 = uv1 = uv2 = glm::vec2(0.0f);
            if (texcoord_accessor_idx != -1) 
            {
                uv0 = *reinterpret_cast<glm::vec2*>(static_cast<uint8_t*>(texcoord_buffer) + texcoord_offset + i0 * texcoord_stride);
                uv1 = *reinterpret_cast<glm::vec2*>(static_cast<uint8_t*>(texcoord_buffer) + texcoord_offset + i1 * texcoord_stride);
                uv2 = *reinterpret_cast<glm::vec2*>(static_cast<uint8_t*>(texcoord_buffer) + texcoord_offset + i2 * texcoord_stride);
            }

            glm::vec4 t0, t1, t2;
            if (tangent_accessor_idx != -1) 
            {
                t0 = *reinterpret_cast<glm::vec4*>(static_cast<uint8_t*>(tangent_buffer) + tangent_offset + i0 * tangent_stride);
                t1 = *reinterpret_cast<glm::vec4*>(static_cast<uint8_t*>(tangent_buffer) + tangent_offset + i1 * tangent_stride);
                t2 = *reinterpret_cast<glm::vec4*>(static_cast<uint8_t*>(tangent_buffer) + tangent_offset + i2 * tangent_stride);
            }

            glm::vec2 bary;
            float t = triangle_intersect(local_ray, v0, v1, v2, bary);
            if (t > 0.0f && t < closest_t)
            {
                closest_t = t;
                inter.t = t;
                glm::vec3 local_normal = bary.x * n1 + bary.y * n2 + (1.0f - bary.x - bary.y) * n0;
                glm::mat3 normal_matrix = glm::transpose(glm::inverse(glm::mat3(node.global_transform)));
                inter.surface_normal = glm::normalize(normal_matrix * local_normal);
                inter.material_id = prim.material_index;
                inter.uv = bary.x * uv1 + bary.y * uv2 + (1.0f - bary.x - bary.y) * uv0;
                inter.tangent = bary.x * t1 + bary.y * t2 + (1.0f - bary.x - bary.y) * t0;
                // Transform tangent to world space
                glm::vec3 local_tangent = glm::vec3(inter.tangent);
                inter.tangent = glm::vec4(glm::normalize(normal_matrix * local_tangent), inter.tangent.w);
            }
        }
    }

    intersections[path_index] = inter;
}

void compute_gltf_intersections(int threads, int num_paths, PathSegments path_segments, pt::glTFModel::DeviceNode* d_nodes, int num_nodes, pt::glTFModel::Primitive* d_primitives, pt::glTFModel::Accessor* d_accessors, pt::glTFModel::BufferView* d_buffer_views, void** d_cu_buffers, ShadeableIntersection* intersections)
{
    dim3 block(threads);
    dim3 grid(divup(num_paths, block.x));
    compute_gltf_intersections_kernel<<<grid, block>>>(num_paths, path_segments, d_nodes, num_nodes, d_primitives, d_accessors, d_buffer_views, d_cu_buffers, intersections);
}

__global__ void final_gather_kernel(int initial_num_paths, glm::vec3* image, PathSegments path_segments)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= initial_num_paths) return;
#ifdef DISABLE_STREAM_COMPACTION
    if (path_segments.remaining_bounces[index] > 0) return;
#endif
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

void shade_paths(int threads, int iteration, int num_paths, ShadeableIntersection* intersections, Material* materials, PathSegments path_segments, cudaTextureObject_t hdri_texture, cudaTextureObject_t* textures, float exposure)
{
    dim3 block(threads);
    dim3 grid(divup(num_paths, block.x));
    shade<<<grid, block>>>(iteration, num_paths, intersections, materials, path_segments, hdri_texture, textures, exposure);
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

__global__ void aces_tonemap_kernel(glm::vec3* input, glm::vec3* output, size_t width, size_t height, float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = x + y * width;
    glm::vec3 color = input[index] * scale;
    // ACES approximation
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    glm::vec3 result = (color * (a * color + b)) / (color * (c * color + d) + e);
    output[index] = glm::clamp(result, 0.0f, 1.0f);
}

void aces_tonemap(const dim3& grid, const dim3& block, glm::vec3* input, glm::vec3* output, size_t width, size_t height, float scale)
{
    aces_tonemap_kernel<<<grid, block>>>(input, output, width, height, scale);
}

__global__ void pbr_neutral_tonemap_kernel(glm::vec3* input, glm::vec3* output, size_t width, size_t height, float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = x + y * width;
    glm::vec3 color = input[index] * scale;

    // https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/pbrNeutral.glsl
    const float start_compression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x_min = glm::min(color.r, glm::min(color.g, color.b));
    float offset = x_min < 0.08f ? x_min - 6.25f * x_min * x_min : 0.04f;
    color -= offset;

    float peak = glm::max(color.r, glm::max(color.g, color.b));
    if (peak < start_compression)
    {
        output[index] = glm::clamp(color, 0.0f, 1.0f);
        return;
    }

    const float d = 1.0f - start_compression;
    const float new_peak = 1.0f - d * d / (peak + d - start_compression);
    color *= new_peak / peak;

    float g = 1.0f - 1.0f / (desaturation * (peak - new_peak) + 1.0f);
    color = glm::mix(color, new_peak * glm::vec3(1.0f, 1.0f, 1.0f), g);
    output[index] = glm::clamp(color, 0.0f, 1.0f);
}

void pbr_neutral_tonemap(const dim3& grid, const dim3& block, glm::vec3* input, glm::vec3* output, size_t width, size_t height, float scale)
{
    pbr_neutral_tonemap_kernel<<<grid, block>>>(input, output, width, height, scale);
}
