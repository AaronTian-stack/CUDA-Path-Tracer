#pragma once
#include "optix_denoiser.h"
#include "../application.h"

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
	PathTracerSettings m_settings;
	std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> m_cmd_bufs{};
protected:
	void init_window() override;
	void create() override;
	void render() override;
	void destroy() override;

public:
	~PathTracer() override;
};
