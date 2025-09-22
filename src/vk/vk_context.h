#pragma once
#include <VkBootstrap.h>

namespace pt
{
	class VulkanContext
	{
		vkb::Instance m_instance;
		vkb::Device m_device;
	public:
		bool create_instance(bool enable_debug_layer);
		bool create_device(VkSurfaceKHR surface);

		friend class VulkanWindow;
	};
}
