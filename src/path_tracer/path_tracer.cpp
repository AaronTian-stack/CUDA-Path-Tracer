#include "path_tracer.h"

#include <csignal>
#include <imgui.h>
#include <imgui_internal.h>
#include <cuda_runtime.h>
#include <SDL3/SDL.h>
#include <glm/gtc/constants.hpp>

#include "bsdf.h"
#include "cuda_pt.h"
#include "util.h"
#include "../vk/vk_cu_interop.h"

void Images::init(size_t num_pixels)
{
	cudaMalloc(&image, num_pixels * sizeof(glm::vec3));
	cudaMemset(image, 0, num_pixels * sizeof(glm::vec3));

	cudaMalloc(&accumulated_albedo, num_pixels * sizeof(glm::vec3));
	cudaMemset(accumulated_albedo, 0, num_pixels * sizeof(glm::vec3));

	cudaMalloc(&accumulated_normal, num_pixels * sizeof(glm::vec3));
	cudaMemset(accumulated_normal, 0, num_pixels * sizeof(glm::vec3));

	cudaMalloc(&albedo, num_pixels * sizeof(glm::vec3));
	cudaMalloc(&normal, num_pixels * sizeof(glm::vec3));
	cudaMalloc(&in_denoise, num_pixels * sizeof(glm::vec3));
	cudaMalloc(&out_denoise, num_pixels * sizeof(glm::vec3));
}

void Images::clear(size_t num_pixels)
{
	cudaMemset(image, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(accumulated_albedo, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(accumulated_normal, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(albedo, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(normal, 0, num_pixels * sizeof(glm::vec3));
}

Images::~Images()
{
	cudaFree(image);
	cudaFree(accumulated_albedo);
	cudaFree(accumulated_normal);
	cudaFree(albedo);
	cudaFree(normal);
	cudaFree(in_denoise);
	cudaFree(out_denoise);
}

void PathTracer::reset_scene()
{
	const auto pixel_count = m_scene.camera.resolution.x * m_scene.camera.resolution.y;
	cudaMemset(m_intersections, 0, pixel_count * sizeof(ShadeableIntersection));
	cudaMemset(m_images.image, 0, pixel_count * sizeof(glm::vec3));
	cudaMemset(m_images.accumulated_albedo, 0, pixel_count * sizeof(glm::vec3));
	cudaMemset(m_images.accumulated_normal, 0, pixel_count * sizeof(glm::vec3));
	cudaMemset(m_images.albedo, 0, pixel_count * sizeof(glm::vec3));
	cudaMemset(m_images.normal, 0, pixel_count * sizeof(glm::vec3));
}

void PathTracer::pathtrace(cudaSurfaceObject_t surf, const PathTracerSettings& settings, const OptiXDenoiser& denoiser,
                           int interval_to_denoise, int iteration)
{
	// Generate primary rays

	const auto& camera = m_scene.camera;

	const auto res_x = camera.resolution.x;
	const auto res_y = camera.resolution.y;
	const auto pixel_count = res_x * res_y;

	const dim3 block_size_2D(16, 16);
	const dim3 blocks_per_grid_2D(
	    (res_x + block_size_2D.x - 1) / block_size_2D.x,
	    (res_y + block_size_2D.y - 1) / block_size_2D.y);

	const int block_size_1D = 128;

	generate_ray_from_camera(blocks_per_grid_2D, block_size_2D, camera, iteration, m_scene_settings.trace_depth, m_paths);

	const dim3 num_blocks_pixels = divup(pixel_count, block_size_1D);

	int depth = 0;
	auto num_paths = pixel_count;

	while (num_paths != 0)
	{
		cudaMemset(m_intersections, 0, pixel_count * sizeof(ShadeableIntersection));

		compute_intersections(block_size_1D, depth, num_paths, m_paths, m_geoms, static_cast<int>(m_scene.geoms.size()), m_intersections);
		
		if (depth++ == 0)
		{
			accumulate_albedo_normal(num_blocks_pixels, block_size_1D,
				pixel_count, m_intersections, m_materials, m_images.accumulated_albedo, m_images.accumulated_normal);
		}

		if (settings.sort_rays)
		{
			sort_paths_by_material(m_intersections, m_paths, num_paths);
		}

		shade_paths(block_size_1D, iteration, num_paths, m_intersections, m_materials, m_paths);

		num_paths = filter_paths_with_bounces(m_paths, num_paths);
	}
	m_settings.traced_depth = depth;

	// Assemble this iteration and apply it to the image
	final_gather(block_size_1D, pixel_count, m_images.image, m_paths);

	normalize_albedo_normal(blocks_per_grid_2D, block_size_2D,
		camera.resolution, iteration, m_images.accumulated_albedo, m_images.accumulated_normal, m_images.albedo, m_images.normal);

	if (iteration % interval_to_denoise == 0)
	{
		//average_image_for_denoise(blocks_per_grid_2D, block_size_2D, m_images.image, camera.resolution, iteration, m_images.in_denoise);
		//denoiser.denoise()
	}

	switch (settings.display_mode)
	{
	case PROGRESSIVE:
		set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.image, res_x, res_y, 1.0f / iteration);
		break;
	case ALBEDO:
		set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.albedo, res_x, res_y, 1.0f);
		break;
	case NORMAL:
		set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.normal, res_x, res_y, 1.0f);
		break;
	case DENOISED:
		set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.out_denoise, res_x, res_y, 1.0f);
		break;
	}
}

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

	// Make all the path tracer resources
	{
		const auto pixel_count = m_scene.camera.resolution.x * m_scene.camera.resolution.y;
		m_images.init(pixel_count);

		cudaMalloc(&m_paths, pixel_count * sizeof(PathSegment));

		cudaMalloc(&m_geoms, m_scene.geoms.size() * sizeof(Geom));
		cudaMemcpy(m_geoms, m_scene.geoms.data(), m_scene.geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

		cudaMalloc(&m_materials, m_scene.materials.size() * sizeof(Material));
		cudaMemcpy(m_materials, m_scene.materials.data(), m_scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

		cudaMalloc(&m_intersections, pixel_count * sizeof(ShadeableIntersection));
		cudaMemset(m_intersections, 0, pixel_count * sizeof(ShadeableIntersection));
	}

	m_context.init_imgui(m_window, m_swapchain);

	m_denoiser.init(800, 800);

	++m_fence_ready_val[get_frame_index()];
}

void PathTracer::render()
{
	static int iteration = 0;

	// TODO: input
	// On input, set iteration = 0

	if (iteration == 0)
	{
		reset_scene();
	}

	// Do CUDA stuff
	{
		//float time = SDL_GetTicks() / 1000.0f;
		//test_set_image(m_interop.surf_obj, 800, 800, time);
		pathtrace(m_interop.surf_obj, m_settings, m_denoiser, denoise_interval, iteration++);
		cudaExternalSemaphoreSignalParams params{};
		cudaSignalExternalSemaphoresAsync(&m_cu_semaphores[m_frame_index], &params, 1, 0);

		// TODO
		if (iteration >= 1)
		{
			// Save image

			exit(EXIT_SUCCESS);
		}
	}

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

	cmd_buf.blitImage(m_texture.image, vk::ImageLayout::eTransferSrcOptimal, m_swapchain.images[swapchain_index], vk::ImageLayout::eTransferDstOptimal, 1, &blit_region, vk::Filter::eNearest);

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
	cudaFree(m_paths);
	cudaFree(m_intersections);
	cudaFree(m_geoms);
	cudaFree(m_materials);

	for (int i = 0; i < m_cuda_semaphores.size(); i++)
	{
		m_context.destroy_cuda_semaphore(&m_cuda_semaphores[i]);
		cudaDestroyExternalSemaphore(m_cu_semaphores[i]);
	}
	free_interop_handles_cuda(&m_interop);
	m_context.destroy_texture(&m_texture);
	m_context.destroy_imgui();
}

bool PathTracer::init_scene(const char* file_name)
{
	const bool result = m_scene.load(file_name, &m_scene_settings);
	assert(m_scene_settings.iterations % denoise_interval == 0);
	return result;
}

PathTracer::~PathTracer()
{
}
