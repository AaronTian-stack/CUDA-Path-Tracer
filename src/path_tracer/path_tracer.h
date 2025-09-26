#pragma once
#include "optix_denoiser.h"
#include "../application.h"
#include <glm/glm.hpp>

#include "scene.h"
#include "scene_structs.h"
#include "../vk/vk_texture.h"

struct Images
{
	glm::vec3* image = nullptr; // Accumulation buffer
	glm::vec3* accumulated_albedo = nullptr;
	glm::vec3* accumulated_normal = nullptr;
	glm::vec3* albedo = nullptr; // Divided by iterations
	glm::vec3* normal = nullptr; // Divided by iterations
	glm::vec3* in_denoise = nullptr;
	glm::vec3* out_denoise = nullptr;
	void init(size_t num_pixels);
	void clear(size_t num_pixels);
	~Images();
};

class PathTracer : public Application
{
	Scene m_scene{};
	OptiXDenoiser m_denoiser{};
	Images m_images{}; // CUDA pointers to row linear 2D arrays

	pt::VulkanTexture m_texture{}; // CUDA is going to write to this texture
	SceneSettings m_scene_settings{};
	pt::InteropHandles m_interop{};
	std::array<CUDASemaphore, MAX_FRAMES_IN_FLIGHT> m_cuda_semaphores{};
	std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> m_cmd_bufs{};
	cudaExternalSemaphore_t m_cu_semaphores[MAX_FRAMES_IN_FLIGHT]{};
	PathTracerSettings m_settings;

	PathSegments m_paths{};
	ShadeableIntersection* m_intersections = nullptr;
	Geom* m_geoms = nullptr;
	Material* m_materials = nullptr;

	SDL_MouseButtonFlags m_mouse_buttons = {};

	static constexpr auto denoise_interval = 10;

	void reset_scene();
	void pathtrace(const PathTracerSettings& settings, const OptiXDenoiser& denoiser, int interval_to_denoise, int iteration);

protected:
	void init_window() override;
	void create() override;
	void render() override;
	void destroy() override;

public:
	bool init_scene(const char* file_name);
	~PathTracer() override;
};
