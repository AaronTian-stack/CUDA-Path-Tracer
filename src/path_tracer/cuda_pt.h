#pragma once
#include <surface_types.h>

#include "optix_denoiser.h"
#include "scene_structs.h"
#include "camera.h"
#include "gltf_model.h"

void test_set_image(cudaSurfaceObject_t surf_obj, size_t width, size_t height, float time);

// Expose wrappers here so that path tracer can use them

void set_image(const dim3& grid, const dim3& block, cudaSurfaceObject_t surf_obj, glm::vec3* image, size_t width, size_t height);
void generate_ray_from_camera(const dim3& grid, const dim3& block, const Camera& cam, int iter, int trace_depth, PathSegments path_segments);
void accumulate_albedo_normal(const dim3& grid, const int block_size_1D,
	int num_paths, ShadeableIntersection* intersections, Material* materials,
	glm::vec3* accumulated_albedo, glm::vec3* accumulated_normal);
void sort_paths_by_material(ShadeableIntersection* intersections, PathSegments path_segments, int num_paths);

void compute_intersections(int threads, int depth, int num_paths, PathSegments path_segments, Geom* geoms, int num_geoms, ShadeableIntersection* intersections);
void compute_gltf_intersections(int threads, int num_paths, PathSegments path_segments, pt::glTFModel::DeviceNode* d_nodes, int num_nodes, pt::glTFModel::Primitive* d_primitives, pt::glTFModel::Accessor* d_accessors, pt::glTFModel::BufferView* d_buffer_views, void** d_cu_buffers, ShadeableIntersection* intersections);
void shade_paths(int threads, int iteration, int num_paths, ShadeableIntersection* intersections, Material* materials, PathSegments path_segments, cudaTextureObject_t hdri_texture, cudaTextureObject_t* textures, float exposure);
int filter_paths_with_bounces(PathSegments path_segments, int num_paths);
void final_gather(int threads, int initial_num_paths, glm::vec3* image, PathSegments path_segments);
void normalize_albedo_normal(const dim3& grid, const dim3& block, glm::vec2 resolution, int iter, glm::vec3* accumulated_albedo, glm::vec3* accumulated_normal, glm::vec3* albedo_image, glm::vec3* normal_image);
void average_image_for_denoise(const dim3& grid, const dim3& block, glm::vec3* image, glm::vec2 resolution, int iter, glm::vec3* in_denoise);
void aces_tonemap(const dim3& grid, const dim3& block, glm::vec3* input, glm::vec3* output, size_t width, size_t height, float scale = 1.0f);
void pbr_neutral_tonemap(const dim3& grid, const dim3& block, glm::vec3* input, glm::vec3* output, size_t width, size_t height, float scale = 1.0f);
void gamma_correct_only(const dim3& grid, const dim3& block, glm::vec3* input, glm::vec3* output, size_t width, size_t height, float scale = 1.0f);
