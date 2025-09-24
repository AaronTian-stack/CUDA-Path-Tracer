#include "vk_cu_interop.h"

#include <cuda_runtime_api.h>

static void free_after_failure(pt::InteropHandles* interop)
{
    cudaFreeMipmappedArray(interop->mip_array);
    cudaDestroyExternalMemory(interop->ext_mem);
    interop->mip_array = nullptr;
    interop->ext_mem = nullptr;
}

bool import_vk_texture_cuda(const pt::VulkanTexture& texture, pt::InteropHandles* interop)
{
    // Make sure these were set properly
    assert(texture.extent.width > 0 && texture.extent.height > 1);
    assert(texture.memory_size > 0);
    assert(texture.win32_handle != nullptr);
        
    const auto width = texture.extent.width;
	const auto height = texture.extent.height;

    cudaExternalMemoryHandleDesc ext_mem_desc{};
    ext_mem_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	ext_mem_desc.handle.win32.handle = texture.win32_handle;
	ext_mem_desc.size = texture.memory_size;

    cudaError_t err = cudaImportExternalMemory(&interop->ext_mem, &ext_mem_desc);
    if (err != cudaSuccess)
    {
        return false;
    }

    assert(texture.format == vk::Format::eR8G8B8A8Unorm);
    cudaExternalMemoryMipmappedArrayDesc desc
    {
		.offset = 0,
		.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned),
		.extent = {width, height, 0}, // Depth is 0 not 1, otherwise interop will not work
		.flags = cudaArraySurfaceLoadStore,
		.numLevels = 1,
        .reserved = {},
    };

    err = cudaExternalMemoryGetMappedMipmappedArray(&interop->mip_array, interop->ext_mem, &desc);
    if (err != cudaSuccess)
    {
        free_after_failure(interop);
        return false;
    }

    err = cudaGetMipmappedArrayLevel(&interop->cu_array, interop->mip_array, 0);
    if (err != cudaSuccess)
    {
        free_after_failure(interop);
        return false;
    }

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = interop->cu_array;

    err = cudaCreateSurfaceObject(&interop->surf_obj, &res_desc);
    if (err != cudaSuccess)
    {
        free_after_failure(interop);
        return false;
    }

    return true;
}

void free_interop_handles_cuda(pt::InteropHandles* interop)
{
    cudaDestroySurfaceObject(interop->surf_obj);
	interop->surf_obj = 0;
    cudaFreeMipmappedArray(interop->mip_array);
    interop->mip_array = nullptr;
    cudaDestroyExternalMemory(interop->ext_mem);
    interop->ext_mem = nullptr;
}

bool import_vk_semaphore_cuda(const CUDASemaphore& semaphore, cudaExternalSemaphore_t* cu_semaphore)
{
    cudaExternalSemaphoreHandleDesc sem_desc{};
	sem_desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	sem_desc.handle.win32.handle = semaphore.handle;
    return cudaImportExternalSemaphore(cu_semaphore, &sem_desc) == cudaSuccess;
}
