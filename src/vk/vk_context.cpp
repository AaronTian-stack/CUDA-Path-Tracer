#include "vk_context.h"

#include <array>

#define VOLK_IMPLEMENTATION
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <volk.h>
#include <vulkan/vulkan.hpp>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include "vk_texture.h"
#include "../application.h"

// This file contains Windows only code for interop and needs to be changed if Linux support is added.
// Unfortunately I don't have a Linux machine and even if I did I don't have the motivation to try it...
// But there are already some preprocessor checks throughout the project to handle some of that logic in case I revisit this in the future.

vk::detail::DispatchLoaderDynamic vk::detail::defaultDispatchLoaderDynamic;

// Require Vulkan 1.3 for dynamic rendering
constexpr auto MAJOR = 1;
constexpr auto MINOR = 3;

bool pt::VulkanContext::create_instance(bool enable_debug_layer)
{
    VkResult result = volkInitialize();
    if (result != VK_SUCCESS)
    {
        return false;
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    vkb::InstanceBuilder builder;
    constexpr auto severity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
    builder.set_app_name("CUDA Path Tracer");

    std::array instance_extensions = 
    {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
    builder.enable_extensions(instance_extensions);

    if (enable_debug_layer)
    {
        builder.request_validation_layers()
            .set_debug_messenger_severity(severity)
            .set_debug_callback(
                [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                    void* pUserData) -> VkBool32
                {
                    OutputDebugStringA(pCallbackData->pMessage);
					OutputDebugStringA("\n");
                    return VK_FALSE;
                });
    }

    auto inst_ret = builder.require_api_version(MAJOR, MINOR).build();
    if (!inst_ret)
    {
        return false;
    }
    m_instance = inst_ret.value();

	volkLoadInstanceOnly(m_instance.instance);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_instance.instance, vkGetInstanceProcAddr);

    return true;
}

bool pt::VulkanContext::create_device(VkSurfaceKHR surface)
{
    if (!surface)
    {
	    // Don't support headless
        return false;
    }

    std::vector<const char*> device_extensions;
    device_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN64
    device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    // TODO: Linux support?
    device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif

    vkb::PhysicalDeviceSelector selector{ m_instance };

    assert(surface);
    assert(MINOR >= 3);

    VkPhysicalDeviceVulkan12Features features12
	{
		.timelineSemaphore = VK_TRUE
	};

    VkPhysicalDeviceVulkan13Features features13
	{
		.synchronization2 = VK_TRUE,
		.dynamicRendering = VK_TRUE,
	};

    auto phys_ret = selector.set_surface(surface)
        .set_minimum_version(MAJOR, MINOR)
        .add_required_extensions(device_extensions)
        .set_required_features_12(features12)
        .set_required_features_13(features13)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
        .select();

    if (!phys_ret)
    {
#ifdef _WIN64
        char buf[512];
        sprintf(buf, "Failed to select physical device: %s\n", phys_ret.error().message().c_str());
        OutputDebugStringA(buf);
#endif
		return false;
    }

    vkb::DeviceBuilder device_builder{ phys_ret.value() };
    auto dev_ret = device_builder.build();
    if (!dev_ret)
    {
#ifdef _WIN64
        char buf[512];
        sprintf(buf, "Failed to create Vulkan device: %s\n", dev_ret.error().message().c_str());
        OutputDebugStringA(buf);
#endif
		return false;
    }

    m_device = dev_ret.value();

	m_physical_device = vk::PhysicalDevice( m_device.physical_device);
	m_logical_device = vk::Device(m_device.device);

    volkLoadDevice(m_device.device);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk::Instance(m_instance.instance), m_logical_device);

    return true;
}

bool pt::VulkanContext::create_swapchain(VulkanSwapchain* const swapchain, VulkanQueue* const queue)
{
    assert(m_surface);
	vkb::SwapchainBuilder swapchain_builder{ m_device, m_surface };
    VkSurfaceFormatKHR desired_format = 
    { 
        VK_FORMAT_B8G8R8A8_UNORM, 
        VK_COLOR_SPACE_SRGB_NONLINEAR_KHR 
    };
    swapchain_builder.set_desired_format(desired_format)
        .set_desired_min_image_count(Application::MAX_FRAMES_IN_FLIGHT)
        .set_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    auto swap_ret = swapchain_builder.build();
    if (!swap_ret) 
    {
#ifdef _WIN64
        char buf[512];
        sprintf(buf, "Failed to create swapchain: %s\n", swap_ret.error().message().c_str());
        OutputDebugStringA(buf);
#endif
        return false;
    }
	swapchain->swapchain = swap_ret.value();

	auto images = swapchain->swapchain.get_images();
    if (!images.has_value())
    {
        return false;
    }
    swapchain->images = std::move(images.value());
	
    auto image_views = swapchain->swapchain.get_image_views();
    if (!image_views.has_value())
    {
        return false;
    }
    swapchain->image_views = std::move(image_views.value());

	m_swapchain_queue = queue;

    return true;
}

bool pt::VulkanContext::create_command_pool(vk::CommandPool* pool, const VulkanQueue& queue) const
{
    vk::CommandPoolCreateInfo pool_info{ .flags = vk::CommandPoolCreateFlagBits::eTransient, .queueFamilyIndex = queue.index };
    const auto result = m_logical_device.createCommandPool(pool_info);
    if (result.result != vk::Result::eSuccess) 
    {
        return false;
    }
    *pool = result.value;
    return true;
}

vk::UniqueCommandPool pt::VulkanContext::create_unique_command_pool(const vk::CommandPool& pool) const
{
    return vk::UniqueCommandPool(pool, m_logical_device);
}

bool pt::VulkanContext::create_command_buffer(const vk::CommandPool& pool, vk::CommandBuffer* cmd_buf) const
{
    vk::CommandBufferAllocateInfo alloc_info{ .commandPool = pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
    const auto result = m_logical_device.allocateCommandBuffers(alloc_info);
    if (result.result != vk::Result::eSuccess) 
    {
        return false;
    }
    *cmd_buf = result.value[0];
    vk::CommandBufferBeginInfo begin_info{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
	return cmd_buf->begin(begin_info) == vk::Result::eSuccess;
}

bool pt::VulkanContext::create_semaphore(vk::Semaphore* semaphore, bool is_timeline, uint64_t initial_value) const
{
    vk::SemaphoreTypeCreateInfo type_info
    {
		.semaphoreType = is_timeline ? vk::SemaphoreType::eTimeline : vk::SemaphoreType::eBinary,
        .initialValue = initial_value,
    };
    vk::SemaphoreCreateInfo semaphore_info{};
    semaphore_info.pNext = &type_info;
    const auto result = m_logical_device.createSemaphore(semaphore_info);
    if (result.result != vk::Result::eSuccess) 
    {
        return false;
    }
    *semaphore = result.value;
    return true;
}

bool pt::VulkanContext::create_cuda_semaphore(CUDASemaphore* semaphore) const
{
    vk::ExportSemaphoreCreateInfo export_info
	{
        .handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32
    };
    vk::SemaphoreCreateInfo sem_info
	{
        .pNext = &export_info
    };
    const auto result = m_logical_device.createSemaphore(sem_info);
    if (result.result != vk::Result::eSuccess) 
    {
        return false;
	}
    auto& cuda_semaphore = result.value;

    vk::SemaphoreGetWin32HandleInfoKHR get_handle_info
	{
        .semaphore = cuda_semaphore,
        .handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32,
    };
    const auto h_result = m_logical_device.getSemaphoreWin32HandleKHR(get_handle_info);
    if (h_result.result != vk::Result::eSuccess)
    {
        m_logical_device.destroySemaphore(cuda_semaphore);
        return false;
    }
	const auto handle = h_result.value;

    *semaphore =
    {
	    .semaphore = cuda_semaphore,
	    .handle = handle,
	};

    return true;
}

void pt::VulkanContext::destroy_cuda_semaphore(CUDASemaphore* semaphore) const
{
    m_logical_device.destroySemaphore(semaphore->semaphore);
    semaphore->semaphore = VK_NULL_HANDLE;
}

vk::UniqueSemaphore pt::VulkanContext::create_unique_semaphore(const vk::Semaphore& semaphore) const
{
	return vk::UniqueSemaphore(semaphore, m_logical_device);
}

bool pt::VulkanContext::create_texture(vk::Format format, vk::Extent2D dimensions, VulkanTexture* texture) const
{
    vk::ExternalMemoryImageCreateInfo externalMemoryInfo
    {
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32,
    };

    vk::ImageCreateInfo image_create_info
    {
			.pNext = &externalMemoryInfo,
            .flags = vk::ImageCreateFlags(),
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = { dimensions.width, dimensions.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
			.tiling = vk::ImageTiling::eOptimal,
			.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive,
    };

    texture->image = m_logical_device.createImage(image_create_info).value;
    vk::MemoryRequirements mem_req = m_logical_device.getImageMemoryRequirements(texture->image);
    texture->memory_size = mem_req.size;

    texture->extent = image_create_info.extent;
    texture->format = format;

    VkExportMemoryAllocateInfo export_info
    {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    };

    vk::PhysicalDeviceMemoryProperties mem_props = m_physical_device.getMemoryProperties();
    uint32_t type_index = 0;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
    {
        if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))
        {
            type_index = i;
            break;
        }
    }

    vk::MemoryAllocateInfo alloc_info
    {
        .pNext = &export_info,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = type_index,
    };

    texture->memory = m_logical_device.allocateMemory(alloc_info).value;
    const auto result = m_logical_device.bindImageMemory(texture->image, texture->memory, 0);
    if (result != vk::Result::eSuccess)
    {
        return false;
	}

    // Export for CUDA interop
	// TODO: Linux support?
    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory = texture->memory;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    texture->win32_handle = nullptr;
    const auto vk_get_memory_win32_handle_khr_fn = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(m_logical_device, "vkGetMemoryWin32HandleKHR"));
    if (vk_get_memory_win32_handle_khr_fn)
    {
        VkResult handleResult = vk_get_memory_win32_handle_khr_fn(m_logical_device, &handleInfo, &texture->win32_handle);
        if (handleResult != VK_SUCCESS)
        {
            texture->win32_handle = nullptr;
            return false;
        }
    }
    else
    {
        return false;
    }

    vk::ImageViewCreateInfo info
    {
        .image = texture->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = 
        {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
		}
    };

    const auto view_result = m_logical_device.createImageView(&info, nullptr, &texture->image_view);
    if (view_result != vk::Result::eSuccess)
    {
        return false;
    }

	return true;
}

void pt::VulkanContext::destroy_texture(VulkanTexture* texture) const
{
    if (texture->image_view)
    {
        m_logical_device.destroyImageView(texture->image_view);
        texture->image_view = VK_NULL_HANDLE;
    }
    if (texture->image)
    {
        m_logical_device.destroyImage(texture->image);
        texture->image = VK_NULL_HANDLE;
    }
    if (texture->memory)
    {
        m_logical_device.freeMemory(texture->memory);
        texture->memory = VK_NULL_HANDLE;
    }
}

bool pt::VulkanContext::present(VulkanSwapchain* swapchain, uint32_t index, uint32_t wait_count, vk::Semaphore* wait_semaphores)
{
    assert(m_swapchain_queue);
	vk::SwapchainKHR swapchain_handle = swapchain->swapchain.swapchain;
    vk::PresentInfoKHR present_info
    {
        .waitSemaphoreCount = wait_count,
        .pWaitSemaphores = wait_semaphores,
        .swapchainCount = 1,
        .pSwapchains = &swapchain_handle,
        .pImageIndices = &index,
    };
    const auto result = m_swapchain_queue->queue.presentKHR(present_info);
	return result == vk::Result::eSuccess;
}

void pt::VulkanContext::free_command_buffers(vk::CommandBuffer* cmd_bufs, uint32_t count,
	const vk::CommandPool& pool) const
{
    m_logical_device.freeCommandBuffers(pool, count, cmd_bufs);
}

bool pt::VulkanContext::end_command_buffer(vk::CommandBuffer* cmd_buf) const
{
    const auto result = cmd_buf->end();
	return result == vk::Result::eSuccess;
}

bool pt::VulkanContext::reset_command_pool(vk::CommandPool* pool)
{
    const auto result = m_logical_device.resetCommandPool(*pool, vk::CommandPoolResetFlagBits::eReleaseResources);
	return result == vk::Result::eSuccess;
}

void pt::VulkanContext::start_render_pass(vk::CommandBuffer* cmd_buf, VulkanSwapchain* swapchain, const uint32_t swapchain_index)
{
    vk::RenderingAttachmentInfo color_attachment_info
	{
		.imageView = swapchain->image_views[swapchain_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
	};

    auto render_area = vk::Rect2D
	{
		.offset = vk::Offset2D{},
		.extent = swapchain->get_extent()
	};
    vk::RenderingInfo render_info
	{
        .renderArea = {render_area},
        .layerCount = 1,
        .viewMask = 0,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_info,
    };

    cmd_buf->beginRendering(render_info);
}

void pt::VulkanContext::end_render_pass(vk::CommandBuffer* cmd_buf)
{
	vkCmdEndRendering(*cmd_buf);
}

void pt::VulkanContext::set_barrier_image(vk::ImageMemoryBarrier* barrier, const VulkanSwapchain& swapchain, unsigned int index)
{
    barrier->image = swapchain.images[index];
    barrier->subresourceRange = 
    {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
	};
}

bool pt::VulkanContext::get_queue(VulkanQueue* queue) const
{
    auto queue_ret = m_device.get_queue(vkb::QueueType::graphics);
    if (!queue_ret) 
    {
#ifdef _WIN64
        char buf[512];
        sprintf(buf, "Failed to get graphics queue: %s\n", queue_ret.error().message().c_str());
        OutputDebugStringA(buf);
#endif
        return false;
    }
    // Should not fail because of get_queue
    const auto index = m_device.get_queue_index(vkb::QueueType::graphics).value(); 
	*queue = 
    {
		.queue = queue_ret.value(),
		.index = index
	};
    return true;
}

uint32_t pt::VulkanContext::get_swapchain_index(const VulkanSwapchain& swapchain, vk::Semaphore* semaphore) const
{
    uint32_t image_index;
    const auto result = vkAcquireNextImageKHR(m_device, swapchain.swapchain, UINT64_MAX, 
        *semaphore, VK_NULL_HANDLE, &image_index);
    if (result != VK_SUCCESS)
    {
        return -1;
    }
    return image_index;
}

uint64_t pt::VulkanContext::get_semaphore_value(const vk::Semaphore& semaphore) const
{
    uint64_t value;
	const auto result = m_logical_device.getSemaphoreCounterValue(semaphore, &value);
    if (result != vk::Result::eSuccess)
    {
		return 0;
	}
	return value;
}

bool pt::VulkanContext::wait_semaphores(const vk::SemaphoreWaitInfo& wait_info)
{
	const auto result = m_logical_device.waitSemaphores(wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess)
    {
		return false;
	}
    return true;
}

void pt::VulkanContext::init_imgui(const VulkanWindow& window, const VulkanSwapchain& swapchain)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    const auto format = swapchain.get_format();
    vk::PipelineRenderingCreateInfo pipeline{ .viewMask = 0, .colorAttachmentCount = 1, .pColorAttachmentFormats = &format };

    assert(m_swapchain_queue);

    ImGui_ImplVulkan_InitInfo init_info
    {
        .ApiVersion = VK_API_VERSION_1_3,
        .Instance = m_instance,
        .PhysicalDevice = m_physical_device,
        .Device = m_logical_device,
        .QueueFamily = m_swapchain_queue->index,
        .Queue = m_swapchain_queue->queue,
        .DescriptorPoolSize = 128,
        .MinImageCount = Application::MAX_FRAMES_IN_FLIGHT,
        .ImageCount = Application::MAX_FRAMES_IN_FLIGHT,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .UseDynamicRendering = true,
        .PipelineRenderingCreateInfo = pipeline,
    };

    ImGui_ImplSDL3_InitForVulkan(window.get_window());
    ImGui_ImplVulkan_Init(&init_info);
}

void pt::VulkanContext::start_imgui_frame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
	ImGui::NewFrame();
}

void pt::VulkanContext::render_imgui_draw_data(vk::CommandBuffer* cmd_buf)
{
    ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd_buf);
}

void pt::VulkanContext::destroy_imgui()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}

bool pt::VulkanContext::wait_idle() const
{
    const auto result = m_logical_device.waitIdle();
	return result == vk::Result::eSuccess;
}

pt::VulkanContext::~VulkanContext()
{
    vkb::destroy_device(m_device);
    vkb::destroy_surface(m_instance, m_surface);
    vkb::destroy_instance(m_instance);
}
