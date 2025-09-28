﻿#include "path_tracer.h"

#include <csignal>
#include <ctime>
#include <imgui.h>
#include <imgui_internal.h>
#include <cuda_runtime.h>
#include <SDL3/SDL.h>
#include <sstream>
#include <stb_image.h>
#include <stb_image_write.h>
#include <string>
#include <vector>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

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
	cudaMalloc(&tonemapped_image, num_pixels * sizeof(glm::vec3));
}

void Images::clear(size_t num_pixels)
{
	cudaMemset(image, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(accumulated_albedo, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(accumulated_normal, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(albedo, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(normal, 0, num_pixels * sizeof(glm::vec3));
	cudaMemset(tonemapped_image, 0, num_pixels * sizeof(glm::vec3));
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
	cudaFree(tonemapped_image);
}

void PathTracer::reset_scene()
{
	const auto pixel_count = m_scene.camera.resolution.x * m_scene.camera.resolution.y;
	m_images.clear(pixel_count);
}

void PathTracer::pathtrace(const PathTracerSettings& settings, const OptiXDenoiser& denoiser,
                           int interval_to_denoise, int iteration)
{
	// Generate primary rays

	const auto& camera = m_scene.camera;

	const auto res_x = camera.resolution.x;
	const auto res_y = camera.resolution.y;
	const auto pixel_count = res_x * res_y;
	
	const dim3 block_size_2D(16, 16);
	const dim3 blocks_per_grid_2D(divup(res_x, block_size_2D.x), divup(res_y, block_size_2D.y));

	const int block_size_1D = 128;

	generate_ray_from_camera(blocks_per_grid_2D, block_size_2D, camera, iteration, m_scene_settings.trace_depth, m_paths);

	const dim3 num_blocks_pixels = divup(pixel_count, block_size_1D);

#ifdef DISABLE_STREAM_COMPACTION
	int depth = 0;
	for (int d = 0; d < m_scene_settings.trace_depth; d++)
	{
		depth = d;
		auto num_paths = pixel_count;

		cudaMemset(m_intersections, 0, pixel_count * sizeof(ShadeableIntersection));

		// Exclusively glTF or default geometry
		if (m_gltf.d_primitives)
		{
			compute_gltf_intersections(block_size_1D, num_paths, m_paths, m_gltf.d_nodes, m_gltf.num_nodes, m_gltf.d_primitives, m_gltf.d_accessors, m_gltf.d_buffer_views, m_gltf.d_buffers, m_intersections);
		}
		else
		{
			compute_intersections(block_size_1D, depth, num_paths, m_paths, m_geoms, static_cast<int>(m_scene.geoms.size()), m_intersections);
		}
		
		if (depth == 0)
		{
			accumulate_albedo_normal(num_blocks_pixels, block_size_1D,
				pixel_count, m_intersections, m_materials, m_images.accumulated_albedo, m_images.accumulated_normal);
		}

		shade_paths(block_size_1D, iteration, num_paths, m_intersections, m_materials, m_paths, m_hdri_texture, m_textures, m_scene_settings.exposure);
	}
	m_settings.traced_depth = depth + 1;
#else
	int depth = 0;
	auto num_paths = pixel_count;

	while (num_paths != 0)
	{
		cudaMemset(m_intersections, 0, pixel_count * sizeof(ShadeableIntersection));

		// Exclusively glTF or default geometry
		if (m_gltf.d_primitives)
		{
			compute_gltf_intersections(block_size_1D, num_paths, m_paths, m_gltf.d_nodes, m_gltf.num_nodes, m_gltf.d_primitives, m_gltf.d_accessors, m_gltf.d_buffer_views, m_gltf.d_buffers, m_intersections);
		}
		else
		{
			compute_intersections(block_size_1D, depth, num_paths, m_paths, m_geoms, static_cast<int>(m_scene.geoms.size()), m_intersections);
		}
		
		if (depth++ == 0)
		{
			accumulate_albedo_normal(num_blocks_pixels, block_size_1D,
				pixel_count, m_intersections, m_materials, m_images.accumulated_albedo, m_images.accumulated_normal);
		}

		if (settings.sort_rays)
		{
			sort_paths_by_material(m_intersections, m_paths, num_paths);
		}

		shade_paths(block_size_1D, iteration, num_paths, m_intersections, m_materials, m_paths, m_hdri_texture, m_textures, m_scene_settings.exposure);

		num_paths = filter_paths_with_bounces(m_paths, num_paths);
	}
	m_settings.traced_depth = depth;
#endif

	// Assemble this iteration and apply it to the image
	final_gather(block_size_1D, pixel_count, m_images.image, m_paths);

	normalize_albedo_normal(blocks_per_grid_2D, block_size_2D,
		camera.resolution, iteration, m_images.accumulated_albedo, m_images.accumulated_normal, m_images.albedo, m_images.normal);

	if ((settings.display_mode == DENOISED && iteration % interval_to_denoise == 0) || iteration >= m_scene_settings.iterations - 1)
	{
		average_image_for_denoise(blocks_per_grid_2D, block_size_2D, m_images.image, camera.resolution, iteration, m_images.in_denoise);

		OptixImage2D in;
		in.data = reinterpret_cast<CUdeviceptr>(m_images.in_denoise);
		in.width = camera.resolution.x;
		in.height = camera.resolution.y;
		in.rowStrideInBytes = sizeof(glm::vec3) * camera.resolution.x;
		in.pixelStrideInBytes = sizeof(glm::vec3);
		in.format = OPTIX_PIXEL_FORMAT_FLOAT3;

		OptixImage2D out = in;
		out.data = reinterpret_cast<CUdeviceptr>(m_images.out_denoise);

		OptixImage2D albedo;
		albedo.data = reinterpret_cast<CUdeviceptr>(m_images.albedo);
		albedo.width = camera.resolution.x;
		albedo.height = camera.resolution.y;
		albedo.rowStrideInBytes = sizeof(glm::vec3) * camera.resolution.x;
		albedo.pixelStrideInBytes = sizeof(glm::vec3);
		albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;

		OptixImage2D normal = albedo;
		normal.data = reinterpret_cast<CUdeviceptr>(m_images.normal);

		THROW_IF_FALSE(denoiser.denoise(in, out, albedo, normal));
	}

	switch (settings.display_mode)
	{
	case PROGRESSIVE:
	{
		glm::vec3* source_image = m_images.image;
		const float scale = 1.0f / static_cast<float>(iteration);
		switch (settings.tonemap_mode)
		{
		case ACES:
			aces_tonemap(blocks_per_grid_2D, block_size_2D, source_image, m_images.tonemapped_image, res_x, res_y, scale);
			set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.tonemapped_image, res_x, res_y, 1.0f);
			break;
		case PBR_NEUTRAL:
			pbr_neutral_tonemap(blocks_per_grid_2D, block_size_2D, source_image, m_images.tonemapped_image, res_x, res_y, scale);
			set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.tonemapped_image, res_x, res_y, 1.0f);
			break;
		case NONE:
			set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, source_image, res_x, res_y, scale);
			break;
		}
		break;
	}
	case ALBEDO:
		set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.albedo, res_x, res_y, 1.0f);
		break;
	case NORMAL:
		set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.normal, res_x, res_y, 1.0f);
		break;
	case DENOISED:
	{
		glm::vec3* source_image = m_images.out_denoise;
		constexpr float scale = 1.0f;
		switch (settings.tonemap_mode)
		{
		case ACES:
			aces_tonemap(blocks_per_grid_2D, block_size_2D, source_image, m_images.tonemapped_image, res_x, res_y, scale);
			set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.tonemapped_image, res_x, res_y, 1.0f);
			break;
		case PBR_NEUTRAL:
			pbr_neutral_tonemap(blocks_per_grid_2D, block_size_2D, source_image, m_images.tonemapped_image, res_x, res_y, scale);
			set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, m_images.tonemapped_image, res_x, res_y, 1.0f);
			break;
		case NONE:
			set_image(blocks_per_grid_2D, block_size_2D, m_interop.surf_obj, source_image, res_x, res_y, scale);
			break;
		}
		break;
	}
	}
}

void PathTracer::init_window()
{
	char title[256] = "CUDA Path Tracer";

	const pt::WindowSettings settings
	{
		.width = m_scene.camera.resolution.x,
		.height = m_scene.camera.resolution.y,
		.title = title,
	};

	THROW_IF_FALSE(m_window.create(settings));
}

void PathTracer::create()
{
	m_context.create_texture(vk::Format::eR8G8B8A8Unorm, {static_cast<uint32_t>(m_scene.camera.resolution.x), static_cast<uint32_t>(m_scene.camera.resolution.y)}, &m_texture);
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

		cudaMalloc(&m_paths.origins, pixel_count * sizeof(glm::vec3));
		cudaMalloc(&m_paths.directions, pixel_count * sizeof(glm::vec3));
		cudaMalloc(&m_paths.colors, pixel_count * sizeof(glm::vec3));
		cudaMalloc(&m_paths.pixel_indices, pixel_count * sizeof(int));
		cudaMalloc(&m_paths.remaining_bounces, pixel_count * sizeof(int));

		cudaMalloc(&m_geoms, m_scene.geoms.size() * sizeof(Geom));
		cudaMemcpy(m_geoms, m_scene.geoms.data(), m_scene.geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

		cudaMalloc(&m_materials, m_scene.materials.size() * sizeof(Material));
		cudaMemcpy(m_materials, m_scene.materials.data(), m_scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

		if (m_gltf.d_primitives)
		{
			cudaFree(m_materials);
			m_materials = m_gltf.d_materials;
			m_textures = m_gltf.d_textures;
		}
		else
		{
			m_textures = nullptr;
		}

		cudaMalloc(&m_intersections, pixel_count * sizeof(ShadeableIntersection));
		cudaMemset(m_intersections, 0, pixel_count * sizeof(ShadeableIntersection));
	}

	if (!m_scene.hdri_path.empty())
	{
		int width, height, channels;
		float* data = stbi_loadf(m_scene.hdri_path.c_str(), &width, &height, &channels, 0);
		if (data)
		{
			const auto channel_desc = cudaCreateChannelDesc<float4>();
			cudaMallocArray(&m_hdri_data, &channel_desc, width, height);

			std::vector<float4> host_data(width * height);
			for (int i = 0; i < width * height; ++i)
			{
				float r = channels >= 1 ? data[i * channels] : 0.0f;
				float g = channels >= 2 ? data[i * channels + 1] : 0.0f;
				float b = channels >= 3 ? data[i * channels + 2] : 0.0f;
				host_data[i] = {.x = r, .y = g, .z = b, .w = 1.0f };
			}

			cudaMemcpy2DToArray(m_hdri_data, 0, 0, host_data.data(), width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyHostToDevice);

			stbi_image_free(data);

			cudaResourceDesc res_desc{};
			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = m_hdri_data;

			cudaTextureDesc tex_desc{};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeElementType;
			tex_desc.normalizedCoords = 1;

			cudaCreateTextureObject(&m_hdri_texture, &res_desc, &tex_desc, nullptr);
		}
#ifdef _WIN64
		else
		{
			char buffer[256];
			snprintf(buffer, sizeof(buffer), "Failed to load HDRI: %s\n", m_scene.hdri_path.c_str());
			OutputDebugStringA(buffer);
		}
#endif
	}

	m_context.init_imgui(m_window, m_swapchain);

	m_denoiser.init(m_scene.camera.resolution.x, m_scene.camera.resolution.y);

	++m_fence_ready_val[get_frame_index()];
}

void PathTracer::render()
{
	static int iteration = 0;

	glm::vec2 mouse_position;
	static glm::vec2 mouse_delta{};
	static glm::vec2 last_mouse_pos{};
	// Reset deltas on transition
	static bool prev_relative_mode = false;

	Uint32 mouse_flags = SDL_GetMouseState(&mouse_position.x, &mouse_position.y);

	const bool left = mouse_flags & SDL_BUTTON_MASK(SDL_BUTTON_LEFT);
	const bool right = mouse_flags & SDL_BUTTON_MASK(SDL_BUTTON_RIGHT);
	const bool middle = mouse_flags & SDL_BUTTON_MASK(SDL_BUTTON_MIDDLE);

	const auto pressed = left || right || middle;
	SDL_SetWindowRelativeMouseMode(m_window.get_window(), pressed);

	const bool relative_mode = SDL_GetWindowRelativeMouseMode(m_window.get_window());

	if (relative_mode)
	{
		glm::vec3 d;
		mouse_flags = SDL_GetRelativeMouseState(&d.x, &d.y);

		if (!prev_relative_mode)
		{
			mouse_delta = {};
		}
		else
		{
			mouse_delta = d;
		}
	}
	else
	{
		if (mouse_flags)
		{
			mouse_delta = mouse_position - last_mouse_pos;
		}
		else
		{
			mouse_delta = {};
		}
	}

	last_mouse_pos = mouse_position;
	prev_relative_mode = relative_mode;

	if (ImGuiIO& io = ImGui::GetIO(); !io.WantCaptureMouse)
	{
		const auto delta = mouse_delta;
		if ((pressed && glm::length2(mouse_delta) > 0))
		{
			iteration = 0;
		}
		if (left)
		{
			auto speed = 0.001f;
			m_scene.camera.rotate_around_target(delta.x * speed, delta.y * speed);
		}
		if (right)
		{
			auto speed = 0.01f;
			m_scene.camera.translate_local(-delta.x * speed, delta.y * speed);
		}
		if (middle)
		{
			float amount = -delta.y * 0.2f;
			auto desired_distance = m_scene.camera.m_target_distance + amount;
			m_scene.camera.set_target_distance(desired_distance);
		}
	}

	static float prev_exposure = m_scene_settings.exposure;
	if (glm::abs(m_scene_settings.exposure - prev_exposure) > 0.01f)
	{
		iteration = 0;
		prev_exposure = m_scene_settings.exposure;
		reset_scene();
	}

	if (iteration == 0)
	{
		reset_scene();
	}

	const auto& camera = m_scene.camera;
	const auto res_x = camera.resolution.x;
	const auto res_y = camera.resolution.y;

	// Do CUDA stuff
	{
		pathtrace(m_settings, m_denoiser, denoise_interval, iteration++);
		
		char title[256];
		snprintf(title, sizeof(title), "CUDA Path Tracer | Iterations: %d", iteration);
		SDL_SetWindowTitle(m_window.get_window(), title);
		
		cudaExternalSemaphoreSignalParams params{};
		cudaSignalExternalSemaphoresAsync(&m_cu_semaphores[m_frame_index], &params, 1, nullptr);

		if (iteration >= m_scene_settings.iterations)
		{
			// Save image
			const auto pixel_count = res_x * res_y;
			const dim3 block_size_2D(16, 16);
			const dim3 blocks_per_grid_2D(divup(res_x, block_size_2D.x), divup(res_y, block_size_2D.y));
			glm::vec3* save_image = m_images.out_denoise;
			switch (m_settings.tonemap_mode)
			{
			case ACES:
				aces_tonemap(blocks_per_grid_2D, block_size_2D, m_images.out_denoise, m_images.tonemapped_image, res_x, res_y, 1.0f);
				save_image = m_images.tonemapped_image;
				break;
			case PBR_NEUTRAL:
				pbr_neutral_tonemap(blocks_per_grid_2D, block_size_2D, m_images.out_denoise, m_images.tonemapped_image, res_x, res_y, 1.0f);
				save_image = m_images.tonemapped_image;
				break;
			case NONE:
				break;
			}
			std::vector<glm::vec3> host_image(pixel_count);
			cudaMemcpy(host_image.data(), save_image, pixel_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

			std::vector<unsigned char> png_data(pixel_count * 3);
			for (size_t i = 0; i < pixel_count; ++i) {
				glm::vec3 color = glm::clamp(host_image[i], 0.0f, 1.0f);
				png_data[i * 3 + 0] = static_cast<unsigned char>(color.r * 255.0f);
				png_data[i * 3 + 1] = static_cast<unsigned char>(color.g * 255.0f);
				png_data[i * 3 + 2] = static_cast<unsigned char>(color.b * 255.0f);
			}
			time_t now;
			time(&now);
			char buf[sizeof "0000-00-00_00-00-00z"];
			strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
			std::string currentTimeString = std::string(buf);
			std::stringstream ss;
			ss << m_scene_settings.output_name << "." << currentTimeString << "." << iteration << "samp.png";
			stbi_write_png(ss.str().c_str(), res_x, res_y, 3, png_data.data(), res_x * 3);
			printf("Image saved to: %s\n", ss.str().c_str());

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
	blit_region.srcOffsets[1] = vk::Offset3D{res_x,res_y,1};
	blit_region.dstSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
	blit_region.dstOffsets[0] = vk::Offset3D{0,0,0};
	blit_region.dstOffsets[1] = vk::Offset3D{res_x,res_y,1};

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

	int tm = m_settings.tonemap_mode;
	ImGui::Combo("Tonemap", &tm, "None\0ACES\0PBR Neutral\0\0");
	m_settings.tonemap_mode = static_cast<TonemapMode>(tm);

	bool has_hdri = m_hdri_texture != 0;
	ImGui::BeginDisabled(!has_hdri);
	ImGui::SliderFloat("HDRI Exposure", &m_scene_settings.exposure, -5.0f, 5.0f);
	ImGui::EndDisabled();

	if (ImGui::DragFloat("Focus Distance", &m_scene.camera.focus_distance, 0.01f))
	{
		iteration = 0;
	}
	if (ImGui::DragFloat("Defocus Angle", &m_scene.camera.defocus_angle, 0.01f))
	{
		iteration = 0;
	}

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
	cudaFree(m_paths.origins);
	cudaFree(m_paths.directions);
	cudaFree(m_paths.colors);
	cudaFree(m_paths.pixel_indices);
	cudaFree(m_paths.remaining_bounces);
	cudaFree(m_intersections);
	cudaFree(m_geoms);
	if (!m_use_gltf_materials)
	{
		cudaFree(m_materials);
	}

	if (m_hdri_texture)
	{
		cudaDestroyTextureObject(m_hdri_texture);
	}
	if (m_hdri_data)
	{
		cudaFreeArray(m_hdri_data);
	}

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
	bool result = m_scene.load(file_name, &m_scene_settings);
	assert(m_scene_settings.iterations % denoise_interval == 0);

	if (!m_scene.gltf_path.empty())
	{
		result = m_gltf.load(m_scene.gltf_path, &m_scene.camera);
	}

	return result;
}

PathTracer::~PathTracer()
{
}
