#pragma once
#include "vk_texture.h"

bool import_vk_texture_cuda(const pt::VulkanTexture& texture, pt::InteropHandles* interop);
void free_interop_handles_cuda(pt::InteropHandles* interop);