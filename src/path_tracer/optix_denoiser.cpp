#include "optix_denoiser.h"

#include <cstdio>
#include <cuda_runtime_api.h>
#include <optix_function_table_definition.h> // Do this only in one cpp file

#ifdef _WIN32
#include <windows.h>
#endif

bool OptiXDenoiser::init(unsigned int width, unsigned int height)
{
    optixInit();
    OptixResult result = optixDeviceContextCreate(nullptr, nullptr, &m_ctx);
    if (result != OPTIX_SUCCESS)
    {
#ifdef _WIN64
        char buf[256];
        sprintf(buf, "OptiX device context creation failed: %d\n", result);
        OutputDebugStringA(buf);
#endif
		return false;
    }

    OptixDenoiserOptions o
    {
        .guideAlbedo = 1,
        .guideNormal = 1,
        .denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY,
    };
    result = optixDenoiserCreate(m_ctx, OPTIX_DENOISER_MODEL_KIND_HDR, &o, &denoiser);
    if (result != OPTIX_SUCCESS)
    {
#ifdef _WIN32
        char buf[256];
        sprintf(buf, "OptiX denoiser creation failed: %d\n", result);
        OutputDebugStringA(buf);
#endif
		return false;
    }

    optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiser_sizes);

    void* state;
    void* scratch;

    cudaMalloc(&state, denoiser_sizes.stateSizeInBytes);
    cudaMalloc(&scratch, denoiser_sizes.withoutOverlapScratchSizeInBytes);

	m_state = reinterpret_cast<CUdeviceptr>(state);
	m_scratch = reinterpret_cast<CUdeviceptr>(scratch);

    return true;
}

bool OptiXDenoiser::denoise(const OptixImage2D& in, const OptixImage2D& out, const OptixImage2D& albedo, const OptixImage2D& normal) const
{
    OptixDenoiserGuideLayer gl{};
    gl.albedo = albedo;
    gl.normal = normal;

    OptixDenoiserLayer ly{};
    ly.input = in;
    ly.output = out;

    OptixDenoiserParams p{};

    const auto denoise_result = optixDenoiserInvoke(denoiser, nullptr, &p, m_state, denoiser_sizes.stateSizeInBytes,
        &gl, &ly, 1, 0, 0, m_scratch, denoiser_sizes.withoutOverlapScratchSizeInBytes);

    if (denoise_result != OPTIX_SUCCESS)
    {
#ifdef _WIN64
        char buf[256];
        sprintf(buf, "OptiX denoiser invoke failed: %d\n", denoise_result);
        OutputDebugStringA(buf);
#endif
        return false;
    }

    return true;
}

OptiXDenoiser::~OptiXDenoiser()
{
    if (m_ctx)
    {
        optixDeviceContextDestroy(m_ctx);
        cudaFree(reinterpret_cast<void*>(m_state));
        cudaFree(reinterpret_cast<void*>(m_scratch));
    }
}
