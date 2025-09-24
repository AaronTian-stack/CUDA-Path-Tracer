#include "vk_window.h"

#include "vk_context.h"

bool pt::VulkanWindow::create(const WindowSettings& settings)
{
	if (!SDL_Init(SDL_INIT_VIDEO))
	{
		SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
		throw std::runtime_error("Unable to initialize SDL");
	}

	SDL_PropertiesID properties_id = SDL_CreateProperties();

	SDL_SetBooleanProperty(properties_id, SDL_PROP_WINDOW_CREATE_VULKAN_BOOLEAN, true);

	SDL_SetNumberProperty(properties_id, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, settings.width);
	SDL_SetNumberProperty(properties_id, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, settings.height);

	SDL_SetNumberProperty(properties_id, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
	SDL_SetNumberProperty(properties_id, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);

	SDL_SetBooleanProperty(properties_id, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, false);

	SDL_SetStringProperty(properties_id, SDL_PROP_WINDOW_CREATE_TITLE_STRING, settings.title);

	m_window = SDL_CreateWindowWithProperties(properties_id);

	if (m_window == nullptr)
	{
		SDL_Log("Unable to create window: %s", SDL_GetError());
		SDL_Quit();
		return false;
	}

	SDL_DestroyProperties(properties_id);

	return true;
}

bool pt::VulkanWindow::create_surface(const VulkanContext& device)
{
	const bool status = SDL_Vulkan_CreateSurface(m_window, device.m_instance, nullptr, &m_surface);
	device.m_surface = m_surface;
	return status;
}
