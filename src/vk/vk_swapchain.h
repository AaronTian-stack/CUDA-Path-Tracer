#pragma once
#include <VkBootstrap.h>

namespace pt
{
	struct VulkanSwapchain
	{
		vkb::Swapchain swapchain;
		std::vector<VkImage> images;
		std::vector<VkImageView> image_views;

		vk::Extent2D get_extent() const { return vk::Extent2D(swapchain.extent.width, swapchain.extent.height); }
		vk::Format get_format() const { return vk::Format(swapchain.image_format); }

		~VulkanSwapchain()
		{
			if (!image_views.empty())
			{
				swapchain.destroy_image_views(image_views);
			}
			vkb::destroy_swapchain(swapchain);
		}
	};
}
