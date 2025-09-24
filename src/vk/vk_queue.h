#pragma once

#include <vulkan/vulkan.hpp>

namespace pt
{
	struct VulkanQueue
	{
		vk::Queue queue;
		unsigned int index;
	};
}
