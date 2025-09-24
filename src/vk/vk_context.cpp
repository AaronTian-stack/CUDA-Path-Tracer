#include "vk_context.h"

#include <array>

#define VOLK_IMPLEMENTATION
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <volk.h>
#include <vulkan/vulkan.hpp>

#ifdef _WIN32
#include <windows.h>
#endif
#include "../application.h"

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

    std::array<const char*, 3> device_extensions = 
    {
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    };

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
#ifdef _WIN32
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
#ifdef _WIN32
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
        .set_desired_min_image_count(Application::MAX_FRAMES_IN_FLIGHT);
    auto swap_ret = swapchain_builder.build();
    if (!swap_ret) 
    {
#ifdef _WIN32
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

vk::UniqueSemaphore pt::VulkanContext::create_unique_semaphore(const vk::Semaphore& semaphore) const
{
	return vk::UniqueSemaphore(semaphore, m_logical_device);
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
        .loadOp = vk::AttachmentLoadOp::eClear,
        .clearValue = vk::ClearValue{ vk::ClearColorValue{ std::array{0.0f, 0.0f, 0.0f, 1.0f} } },
	};

    auto render_area = vk::Rect2D
	{
		vk::Offset2D{},
		swapchain->get_extent()
	};
	const auto format = swapchain->get_format();
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
#ifdef _WIN32
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
#ifdef _WIN32
        char buf[256];
        sprintf(buf, "Failed to acquire next image: %d\n", result);
        OutputDebugStringA(buf);
#endif
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
#ifdef _WIN32
        char buf[256];
        sprintf(buf, "Failed to get semaphore value: %d\n", result);
        OutputDebugStringA(buf);
#endif
		return 0;
	}
	return value;
}

bool pt::VulkanContext::wait_fences(const vk::SemaphoreWaitInfo& wait_info)
{
	const auto result = m_logical_device.waitSemaphores(wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess)
    {
#ifdef _WIN32
        char buf[256];
        sprintf(buf, "Failed to wait for semaphores: %d\n", result);
        OutputDebugStringA(buf);
#endif
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
        .MinImageCount = 2,
        .ImageCount = 2,
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
