#pragma once
#include "optix_denoiser.h"
#include "../application.h"
#include <glm/glm.hpp>

#include "../vk/vk_texture.h"

enum DisplayMode : uint8_t
{
	PROGRESSIVE,
	ALBEDO,
	NORMAL,
	DENOISED,
};

struct PathTracerSettings
{
	int traced_depth = 0;
	bool sort_rays = false;
	DisplayMode display_mode = PROGRESSIVE;
};

class PathTracer : public Application
{
	OptiXDenoiser m_denoiser;

	struct Images
	{
		glm::vec3* image = nullptr; // Accumulation buffer
		glm::vec3* accumulated_albedo = nullptr;
		glm::vec3* accumulated_normal = nullptr;
		glm::vec3* albedo = nullptr; // Divided by iterations
		glm::vec3* normal = nullptr; // Divided by iterations
		glm::vec3* in_denoise = nullptr;
		glm::vec3* out_denoise = nullptr;
	} m_images{}; // CUDA pointers to row linear 2D arrays

	pt::VulkanTexture m_texture{}; // CUDA is going to write to this texture
	pt::InteropHandles m_interop{};
	std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> m_cmd_bufs{};
	PathTracerSettings m_settings;

protected:
	void init_window() override;
	void create() override;
	void render() override;
	void destroy() override;

public:
	~PathTracer() override;
};
