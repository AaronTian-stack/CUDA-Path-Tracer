#pragma once
#include <stdexcept>

#include "vk/vk_context.h"
#include "vk/vk_window.h"
#include "vk/vk_queue.h"
#include "vk/vk_swapchain.h"

#define THROW_IF_FALSE(cond) do { if (!(cond)) throw std::runtime_error(#cond " failed"); } while (0)

// Partially adapted from my own open source project QhenkiX
// Self plug: https://github.com/AaronTian-stack/QhenkiX
class Application
{
public:
	static constexpr unsigned int MAX_FRAMES_IN_FLIGHT = 2;
	unsigned int m_frame_index = 0;

protected:
	pt::VulkanContext m_context;
	pt::VulkanWindow m_window;
	pt::VulkanSwapchain m_swapchain;
	pt::VulkanQueue m_queue;
	vk::UniqueCommandPool m_cmd_pools[MAX_FRAMES_IN_FLIGHT];
	vk::UniqueSemaphore m_image_available_semaphores[MAX_FRAMES_IN_FLIGHT];
	vk::UniqueSemaphore m_render_finished_semaphores[MAX_FRAMES_IN_FLIGHT];
	vk::UniqueSemaphore m_fence;
	std::array<uint64_t, MAX_FRAMES_IN_FLIGHT> m_fence_ready_val = {};

	bool m_quit = false;

	virtual void init_window();
	virtual void create() {}
	virtual void render() {}
	virtual void destroy() {}

public:
	unsigned int get_frame_index() { return m_frame_index; }
	void increment_frame_index() { m_frame_index = (m_frame_index + 1) % MAX_FRAMES_IN_FLIGHT; }

	void run(bool enable_debug_layer);
	virtual ~Application() = 0;
};
