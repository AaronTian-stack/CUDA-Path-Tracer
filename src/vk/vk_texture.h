#pragma once
#include <driver_types.h>
#include <surface_types.h>
#include <VkBootstrap.h>
#include <vulkan/vulkan.hpp>

namespace pt
{
	struct VulkanTexture
	{
		vk::Image image;
		vk::ImageView image_view;
		vk::Format format;
		VkExtent3D extent;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		size_t memory_size = 0;
		HANDLE win32_handle = nullptr; // For CUDA
	};
	struct InteropHandles
	{
		cudaExternalMemory_t ext_mem = nullptr;
		cudaMipmappedArray_t mip_array = nullptr;
		cudaArray_t cu_array = nullptr;
		cudaSurfaceObject_t surf_obj = 0;
	};
}