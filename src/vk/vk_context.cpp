#include "vk_context.h"
#include <vulkan/vulkan.h>
#include <array>

// Require Vulkan 1.3 for dynamic rendering
constexpr auto MAJOR = 1;
constexpr auto MINOR = 3;

bool pt::VulkanContext::create_instance(bool enable_debug_layer)
{
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
                    fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);
                    return VK_FALSE;
                });
    }

    auto inst_ret = builder.require_api_version(MAJOR, MINOR).build();
    if (!inst_ret)
    {
        fprintf(stderr, "Failed to create Vulkan instance: %s\n", inst_ret.error().message().c_str());
        return false;
    }
    m_instance = inst_ret.value();
    return true;
}

bool pt::VulkanContext::create_device(VkSurfaceKHR surface)
{
    if (surface)
    {
	    // Don't support headless
        if (!surface)
        {
            fprintf(stderr, "Surface required\n");
            return false;
        }
    }

    std::array<const char*, 1> device_extensions = 
    {
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };

    vkb::PhysicalDeviceSelector selector{ m_instance };

    assert(surface);

    auto phys_ret = selector.set_surface(surface)
        .set_minimum_version(MAJOR, MINOR)
        .add_required_extensions(device_extensions)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
        .select();

    if (!phys_ret)
    {
        fprintf(stderr, "Failed to select physical device: %s\n", phys_ret.error().message().c_str());
		return false;
    }

    vkb::DeviceBuilder device_builder{ phys_ret.value() };
    auto dev_ret = device_builder.build();
    if (!dev_ret)
    {
		fprintf(stderr, "Failed to create Vulkan device: %s\n", dev_ret.error().message().c_str());
		return false;
    }

    m_device = dev_ret.value();

    return true;
}
