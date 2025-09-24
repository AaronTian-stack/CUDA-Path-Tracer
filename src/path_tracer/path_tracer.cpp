#include "path_tracer.h"

#include <csignal>
#include <imgui.h>
#include <imgui_internal.h>
#include <cuda_runtime.h>
#include <SDL3/SDL.h>

#include "cuda_pt.h"
#include "../vk/vk_cu_interop.h"

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
	m_context.create_texture(vk::Format::eR8G8B8A8Unorm, {800, 800}, &m_texture);
	THROW_IF_FALSE(import_vk_texture_cuda(m_texture, &m_interop));

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		THROW_IF_FALSE(m_context.create_cuda_semaphore(&m_cuda_semaphores[i]));
		THROW_IF_FALSE(import_vk_semaphore_cuda(m_cuda_semaphores[i], &m_cu_semaphores[i]));
	}

	// TODO: make all the path tracer resources

	m_context.init_imgui(m_window, m_swapchain);

	m_denoiser.init(800, 800);

	++m_fence_ready_val[get_frame_index()];
}

void PathTracer::render()
{
	// Do CUDA stuff
	float time = SDL_GetTicks() / 1000.0f;
	test_set_image(m_interop.surf_obj, 800, 800, time, m_cu_semaphores[m_frame_index]);

	const auto swapchain_index = m_context.get_swapchain_index(m_swapchain, &m_image_available_semaphores[m_frame_index].get());
	assert(swapchain_index != -1);

	m_context.reset_command_pool(&m_cmd_pools[m_frame_index].get());

	auto& cmd_buf = m_cmd_bufs[get_frame_index()];
	// Required otherwise memory will grow
	m_context.free_command_buffers(&cmd_buf, 1, m_cmd_pools[get_frame_index()].get());
	m_context.create_command_buffer(m_cmd_pools[m_frame_index].get(), &cmd_buf);

	vk::ImageMemoryBarrier texture_barrier
	{
		.srcAccessMask = vk::AccessFlagBits::eNone,
		.dstAccessMask = vk::AccessFlagBits::eTransferRead,
		.oldLayout = vk::ImageLayout::eGeneral,
		.newLayout = vk::ImageLayout::eTransferSrcOptimal,
		.image = m_texture.image,
		.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
	};

	vk::ImageMemoryBarrier barrier_render
	{
		.srcAccessMask = vk::AccessFlagBits::eNone,
		.dstAccessMask = vk::AccessFlagBits::eTransferWrite,
		.oldLayout = vk::ImageLayout::eUndefined,
		.newLayout = vk::ImageLayout::eTransferDstOptimal,
	};
	m_context.set_barrier_image(&barrier_render, m_swapchain, swapchain_index);

	std::array barriers = { texture_barrier, barrier_render };

	cmd_buf.pipelineBarrier(
		vk::PipelineStageFlagBits::eTopOfPipe,
		vk::PipelineStageFlagBits::eTransfer,
		vk::DependencyFlags(),
		0, nullptr,
		0, nullptr,
		2, barriers.data()
	);

	vk::ImageBlit blit_region
	{
		.srcSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
	};
	blit_region.srcOffsets[0] = vk::Offset3D{0,0,0};
	blit_region.srcOffsets[1] = vk::Offset3D{800,800,1};
	blit_region.dstSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
	blit_region.dstOffsets[0] = vk::Offset3D{0,0,0};
	blit_region.dstOffsets[1] = vk::Offset3D{800,800,1};

	cmd_buf.blitImage(m_texture.image, vk::ImageLayout::eTransferSrcOptimal, m_swapchain.images[swapchain_index], vk::ImageLayout::eTransferDstOptimal, 1, &blit_region, vk::Filter::eLinear);

	vk::ImageMemoryBarrier color_barrier
	{
		.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
		.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead,
		.oldLayout = vk::ImageLayout::eTransferDstOptimal,
		.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	m_context.set_barrier_image(&color_barrier, m_swapchain, swapchain_index);

	cmd_buf.pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer,
		vk::PipelineStageFlagBits::eColorAttachmentOutput,
		vk::DependencyFlags(),
		0, nullptr,
		0, nullptr,
		1, &color_barrier
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
		.srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead,
		.dstAccessMask = vk::AccessFlagBits::eNone,
		.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.newLayout = vk::ImageLayout::ePresentSrcKHR,
	};
	m_context.set_barrier_image(&present_barrier, m_swapchain, swapchain_index);

	cmd_buf.pipelineBarrier(
		vk::PipelineStageFlagBits::eColorAttachmentOutput,
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
		// Wait for image to be available and CUDA to finish
		std::array<vk::SemaphoreSubmitInfo, 2> wait_infos;
		wait_infos[0] = {
			.semaphore = m_image_available_semaphores[m_frame_index].get(),
			.stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		};
		wait_infos[1] = {
			.semaphore = m_cuda_semaphores[m_frame_index].semaphore,
			.stageMask = vk::PipelineStageFlagBits2::eTransfer,
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
			.waitSemaphoreInfoCount = 2,
			.pWaitSemaphoreInfos = wait_infos.data(),
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
		m_context.wait_semaphores(wait_info);
	}
	m_fence_ready_val[get_frame_index()] = current_fence_value + 1;
}

void PathTracer::destroy()
{
	for (int i = 0; i < m_cuda_semaphores.size(); i++)
	{
		m_context.destroy_cuda_semaphore(&m_cuda_semaphores[i]);
		cudaDestroyExternalSemaphore(m_cu_semaphores[i]);
	}
	free_interop_handles_cuda(&m_interop);
	m_context.destroy_texture(&m_texture);
	m_context.destroy_imgui();
}

PathTracer::~PathTracer()
{
}
