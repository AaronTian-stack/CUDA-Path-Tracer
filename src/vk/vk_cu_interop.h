#pragma once
#include "vk_texture.h"

struct CUDASemaphore
{
	vk::Semaphore semaphore;
	HANDLE handle;
};

bool import_vk_texture_cuda(const pt::VulkanTexture& texture, pt::InteropHandles* interop);
void free_interop_handles_cuda(pt::InteropHandles* interop);

bool import_vk_semaphore_cuda(const CUDASemaphore& semaphore, cudaExternalSemaphore_t* cu_semaphore);
