#pragma once

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>

#include "vk_context.h"

namespace pt
{
	struct WindowSettings
	{
		int64_t width, height;
		const char* title;
	};

	class VulkanWindow
	{
		SDL_Window* m_window = nullptr;
		VkSurfaceKHR m_surface = VK_NULL_HANDLE;
	public:
		bool create(const WindowSettings& settings);
		bool create_surface(const VulkanContext& device);
		VkSurfaceKHR get_surface() const { return m_surface; }
	};
}
