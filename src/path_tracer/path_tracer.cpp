#include "path_tracer.h"

#include <csignal>
#include <imgui.h>
#include <imgui_internal.h>

void PathTracer::init_window()
{
	char title[256] = "CUDA Path Tracer";

	// TODO: change based off command line settings
	const pt::WindowSettings settings
	{
		.width = 800,
		.height = 800,
		.title = title,
	};

	THROW_IF_FALSE(m_window.create(settings));
}

void PathTracer::create()
{
	// TODO: create texture to render to

	// TODO: make all the path tracer resources

	m_context.init_imgui(m_window, m_swapchain);

	m_denoiser.init(800, 800);

	++m_fence_ready_val[get_frame_index()];
}

void PathTracer::render()
{
	// Do CUDA stuff

	const auto swapchain_index = m_context.get_swapchain_index(m_swapchain, &m_image_available_semaphores[m_frame_index].get());
	assert(swapchain_index != -1);

	m_context.reset_command_pool(&m_cmd_pools[m_frame_index].get());

	auto& cmd_buf = m_cmd_bufs[get_frame_index()];
	// Required otherwise memory will grow
	m_context.free_command_buffers(&cmd_buf, 1, m_cmd_pools[get_frame_index()].get());
	m_context.create_command_buffer(m_cmd_pools[m_frame_index].get(), &cmd_buf);

	vk::ImageMemoryBarrier barrier_render
	{
		.oldLayout = vk::ImageLayout::eUndefined,
		.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	m_context.set_barrier_image(&barrier_render, m_swapchain, swapchain_index);

	cmd_buf.pipelineBarrier(
		vk::PipelineStageFlagBits::eTopOfPipe,
		vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlags(),
		0, nullptr,
		0, nullptr,
		1, &barrier_render
	);

	m_context.start_imgui_frame();

	ImGui::Begin("Path Tracer Analytics");

	if (ImGui::GetCurrentWindow()->Appearing)
	{
		ImGui::SetWindowSize(ImVec2(0, 0));
	}
	ImGui::Text("Traced Depth %d", m_settings.traced_depth);
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Checkbox("Sort Rays", &m_settings.sort_rays);

	int t = m_settings.display_mode;
	ImGui::Combo("Display Mode", &t, "Progressive\0Albedo\0Normals\0Denoised\0\0");
	m_settings.display_mode = static_cast<DisplayMode>(t);

	ImGui::End();

	m_context.start_render_pass(&cmd_buf, &m_swapchain, swapchain_index);
	// TODO: blit texture
	m_context.render_imgui_draw_data(&cmd_buf);
	m_context.end_render_pass(&cmd_buf);

	vk::ImageMemoryBarrier present_barrier
	{
		.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.newLayout = vk::ImageLayout::ePresentSrcKHR,
	};
	m_context.set_barrier_image(&present_barrier, m_swapchain, swapchain_index);

	cmd_buf.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands,
		vk::PipelineStageFlagBits::eBottomOfPipe,
		vk::DependencyFlags(),
		0, nullptr,
		0, nullptr,
		1, &present_barrier
	);

	m_context.end_command_buffer(&cmd_buf);

	auto current_fence_value = m_fence_ready_val[m_frame_index];
	{
		vk::CommandBufferSubmitInfo cmd_buf_info
		{
			.commandBuffer = cmd_buf,
		};
		// Wait for image to be available
		vk::SemaphoreSubmitInfo submit_info
		{
			.semaphore = m_image_available_semaphores[m_frame_index].get(),
			.stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		};
		// Signal render finished
		std::array<vk::SemaphoreSubmitInfo, 2> signal_infos;
		// Timeline
		signal_infos[0] =
		{
			.semaphore = m_fence.get(),
			.value = current_fence_value,
			.stageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
		};
		// Binary
		signal_infos[1] =
		{
			.semaphore = m_render_finished_semaphores[m_frame_index].get(),
			.stageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
		};
		vk::SubmitInfo2 info
		{
			.waitSemaphoreInfoCount = 1,
			.pWaitSemaphoreInfos = &submit_info,
			.commandBufferInfoCount = 1,
			.pCommandBufferInfos = &cmd_buf_info,
			.signalSemaphoreInfoCount = 2,
			.pSignalSemaphoreInfos = signal_infos.data(),
		};
		const auto result = m_queue.queue.submit2(info, VK_NULL_HANDLE);
		assert(result == vk::Result::eSuccess);
	}

	m_context.present(&m_swapchain, swapchain_index, 1, &m_render_finished_semaphores[m_frame_index].get());

	increment_frame_index();

	// Wait if frame is not ready
	if (m_context.get_semaphore_value(m_fence.get()) < m_fence_ready_val[get_frame_index()])
	{
		vk::SemaphoreWaitInfo wait_info
		{
			.semaphoreCount = 1,
			.pSemaphores = &m_fence.get(),
			.pValues = &current_fence_value,
		};
		m_context.wait_fences(wait_info);
	}
	m_fence_ready_val[get_frame_index()] = current_fence_value + 1;
}

void PathTracer::destroy()
{
	m_context.destroy_imgui();
}

PathTracer::~PathTracer()
{
}
