#pragma once
#include <optix_stubs.h>

class OptiXDenoiser
{
	OptixDeviceContext m_ctx = nullptr;
	OptixDenoiser denoiser;
	OptixDenoiserSizes denoiser_sizes{};
	CUdeviceptr m_state = NULL;
	CUdeviceptr m_scratch = NULL;
	unsigned int m_width, m_height;
public:
	OptiXDenoiser() = default;
	bool init(unsigned int width, unsigned int height);
	bool denoise(const OptixImage2D& in, const OptixImage2D& out, const OptixImage2D& albedo, const OptixImage2D& normal, unsigned int width, unsigned int height);
	~OptiXDenoiser();
};
