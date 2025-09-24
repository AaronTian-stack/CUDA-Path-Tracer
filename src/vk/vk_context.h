#pragma once
#include <VkBootstrap.h>

#include "vk_queue.h"
#include "vk_swapchain.h"
#include "vk_window.h"

namespace pt
{
	class VulkanContext
	{
		vkb::Instance m_instance;
		vkb::Device m_device;
		// Compability with vulkan.hpp
		vk::PhysicalDevice m_physical_device = VK_NULL_HANDLE;
		vk::Device m_logical_device = VK_NULL_HANDLE;
		VulkanQueue* m_swapchain_queue;
		mutable VkSurfaceKHR m_surface = VK_NULL_HANDLE;
	public:
		bool create_instance(bool enable_debug_layer);
		bool create_device(VkSurfaceKHR surface);
		bool create_swapchain(VulkanSwapchain* swapchain, VulkanQueue* queue);
		bool create_command_pool(vk::CommandPool* pool, const VulkanQueue& queue) const;
		vk::UniqueCommandPool create_unique_command_pool(const vk::CommandPool& pool) const;
		bool create_command_buffer(const vk::CommandPool& pool, vk::CommandBuffer* cmd_buf) const;
		bool create_semaphore(vk::Semaphore* semaphore, bool is_timeline, uint64_t initial_value) const;
		vk::UniqueSemaphore create_unique_semaphore(const vk::Semaphore& semaphore) const;

		bool present(VulkanSwapchain* swapchain, uint32_t index, uint32_t wait_count, vk::Semaphore* wait_semaphores);

		void free_command_buffers(vk::CommandBuffer* cmd_bufs, uint32_t count, const vk::CommandPool& pool) const;
		bool end_command_buffer(vk::CommandBuffer* cmd_buf) const;

		bool reset_command_pool(vk::CommandPool* pool);

		void start_render_pass(vk::CommandBuffer* cmd_buf, VulkanSwapchain* swapchain, uint32_t swapchain_index);
		void end_render_pass(vk::CommandBuffer* cmd_buf);

		void set_barrier_image(vk::ImageMemoryBarrier* barrier, const VulkanSwapchain& swapchain, unsigned int index);

		vk::Instance get_instance() const { return m_instance.instance; }
		bool get_queue(VulkanQueue* queue) const;
		uint32_t get_swapchain_index(const VulkanSwapchain& swapchain, vk::Semaphore* semaphore) const;
		uint64_t get_semaphore_value(const vk::Semaphore& semaphore) const;
		bool wait_fences(const vk::SemaphoreWaitInfo& wait_info);

		void init_imgui(const VulkanWindow& window, const VulkanSwapchain& swapchain);
		void start_imgui_frame();
		void render_imgui_draw_data(vk::CommandBuffer* cmd_buf);
		void destroy_imgui();

		bool wait_idle() const;

		~VulkanContext();

		friend class VulkanWindow;
	};
}
