#include "application.h"

#include <imgui_impl_sdl3.h>

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
	assert(MAX_FRAMES_IN_FLIGHT <= 2);
	init_window();

	THROW_IF_FALSE(m_context.create_instance(enable_debug_layer));
	THROW_IF_FALSE(m_window.create_surface(m_context));
	THROW_IF_FALSE(m_context.create_device(m_window.get_surface()));

	THROW_IF_FALSE(m_context.get_queue(&m_queue));

	THROW_IF_FALSE(m_context.create_swapchain(&m_swapchain, &m_queue));

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::CommandPool pool;
		THROW_IF_FALSE(m_context.create_command_pool(&pool, m_queue));
		m_cmd_pools[i] = m_context.create_unique_command_pool(pool);
	}

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::Semaphore img_avail;
		THROW_IF_FALSE(m_context.create_semaphore(&img_avail, false, 0));
		m_image_available_semaphores[i] = m_context.create_unique_semaphore(img_avail);

		vk::Semaphore render_finished;
		THROW_IF_FALSE(m_context.create_semaphore(&render_finished, false, 0));
		m_render_finished_semaphores[i] = m_context.create_unique_semaphore(render_finished);
	}

	vk::Semaphore fence;
	THROW_IF_FALSE(m_context.create_semaphore(&fence, true, 0));
	m_fence = m_context.create_unique_semaphore(fence);

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
			if (ImGui::GetCurrentContext())
				ImGui_ImplSDL3_ProcessEvent(&event);
		}
		render();
	}
	THROW_IF_FALSE(m_context.wait_idle());
	destroy();
}

Application::~Application() = default;
