#include "application.h"

void Application::init_window()
{
	char title[256] = "Vulkan App";

	const pt::WindowSettings settings
	{
		.width = 800,
		.height = 800,
		.title = title,
	};

	THROW_IF_FALSE(m_window.create(settings));
}

void Application::run(bool enable_debug_layer)
{
	init_window();

	THROW_IF_FALSE(m_context.create_instance(enable_debug_layer));
	THROW_IF_FALSE(m_window.create_surface(m_context));
	THROW_IF_FALSE(m_context.create_device(m_window.get_surface()));

	create();

	while (!m_quit)
	{
		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_EVENT_QUIT)
			{
				m_quit = true;
			}
			// TODO: input
			render();
		}
		// TODO: wait idle
		destroy();
	}
}

Application::~Application() = default;
