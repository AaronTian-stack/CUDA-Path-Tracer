#include "pathtracer.h"

#include <imgui.h>

void PathTracer::create()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	// TODO: init imgui for vulkan

	// TODO: make all the path tracer resources
}

void PathTracer::render()
{
	// TODO: start imgui frame

	// Create command list

	// Transition swapchain image

	// Do CUDA stuff

	// Blit to swapchain image

	// Close command list

	// Wait for previous frame
}

void PathTracer::destroy()
{
	
}

PathTracer::~PathTracer()
{
}
