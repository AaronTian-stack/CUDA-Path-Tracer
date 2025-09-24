#pragma once

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan_core.h>

namespace pt
{
	struct WindowSettings
	{
		int64_t width, height;
		const char* title;
	};

	class VulkanContext;

	class VulkanWindow
	{
		SDL_Window* m_window = nullptr;
		// Created by window but managed by VulkanContext
		VkSurfaceKHR m_surface = VK_NULL_HANDLE;
	public:
		bool create(const WindowSettings& settings);
		bool create_surface(const VulkanContext& device);
		SDL_Window* get_window() const { return m_window; }
		VkSurfaceKHR get_surface() const { return m_surface; }
	};
}
