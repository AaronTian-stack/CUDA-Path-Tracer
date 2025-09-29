#pragma once

#include <glm/glm.hpp>

#include "util.h"

// GGX model as described in https://seblagarde.wordpress.com/wp-content/uploads/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
// All the below functions come from the above SIGGRAPH course

__device__ glm::vec3 f_schlick(glm::vec3 f0, float f90, float u)
{
	return f0 + (f90 - f0) * glm::pow(1.0f - u, 5.0f);
}

__device__ float fr_disney_diffuse(float NdotV, float NdotL, float LdotH, float linear_roughness)
{
	const float energy_bias = glm::mix(0.0f, 0.5f, linear_roughness);
	const float energy_factor = glm::mix(1.0f, 1.0f / 1.51f, linear_roughness);
	const float fd90 = energy_bias + 2.0f * LdotH * LdotH * linear_roughness;
	const glm::vec3 f0 = glm::vec3(1.0f);
	const float light_scatter = f_schlick(f0, fd90, NdotL).r;
	const float view_scatter = f_schlick(f0, fd90, NdotV).r;

	return light_scatter * view_scatter * energy_factor;
}

__device__ float v_smith_ggx_correlated(float NdotL, float NdotV, const float alphaG)
{
	float alpha_g2 = alphaG * alphaG;
	alpha_g2 = alpha_g2 + 0.0000001f;
	float lambda_ggxv = NdotL * sqrt((-NdotV * alpha_g2 + NdotV) * NdotV + alpha_g2);
	float lambda_ggxl = NdotV * sqrt((-NdotL * alpha_g2 + NdotL) * NdotL + alpha_g2);

	return 1.0f / (1.0f + lambda_ggxv + lambda_ggxl);
}

__device__ float d_ggx(float NdotH, float m)
{
	const float m2 = m * m;
	const float f = (NdotH * m2 - NdotH) * NdotH + 1;
	return m2 / (f * f) / PI;
}

__device__ glm::vec3 get_f(float LdotH, glm::vec3 f0)
{
	const float f90 = glm::clamp(50.0f * glm::dot(f0, glm::vec3(0.33f)), 0.0f, 1.0f);
	return f_schlick(f0, f90, LdotH);
}

__device__ glm::vec3 get_specular(float NdotV, float NdotL, float LdotH, float NdotH, float roughness, glm::vec3 f0, glm::vec3* F)
{
	*F = get_f(LdotH, f0);
	const float G = v_smith_ggx_correlated(NdotV, NdotL, roughness);
	const float D = d_ggx(NdotH, roughness);
	const glm::vec3 Fr = D * (*F) * G / (4.0f * NdotV * NdotL);
	return Fr;
}


// Microfacet sampling functions adapted from CIS 5610 / Pbrt

__device__ float cos_theta(glm::vec3 w)
{
	return w.z;
}
__device__ float cos2_theta(glm::vec3 w)
{
	return w.z * w.z;
}
__device__ float abs_cos_theta(glm::vec3 w)
{
	return glm::abs(w.z);
}
__device__ float sin2_theta(glm::vec3 w)
{
	return max(0.0f, 1.0f - cos2_theta(w));
}
__device__ float sin_theta(glm::vec3 w)
{
	return glm::sqrt(sin2_theta(w));
}
__device__ float tan_theta(glm::vec3 w)
{
	return sin_theta(w) / cos_theta(w);
}
__device__ float tan2_theta(glm::vec3 w)
{
    return sin2_theta(w) / cos2_theta(w);
}
__device__ float cos_phi(glm::vec3 w)
{
    const float st = sin_theta(w);
    return st == 0.0f ? 1.0f : glm::clamp(w.x / st, -1.0f, 1.0f);
}
__device__ float sin_phi(glm::vec3 w)
{
    const float st = sin_theta(w);
    return st == 0.0f ? 0.0f : glm::clamp(w.y / st, -1.0f, 1.0f);
}
__device__ float cos2_phi(glm::vec3 w)
{
	return cos_phi(w) * cos_phi(w);
}
__device__ float sin2_phi(glm::vec3 w)
{
	return sin_phi(w) * sin_phi(w);
}

__device__ float trowbridge_reitz_d(glm::vec3 wh, float roughness)
{
	const float t2t = tan2_theta(wh);
	if (isinf(t2t))
	{
		return 0.0f;
	}

	const float cos4_theta = cos2_theta(wh) * cos2_theta(wh);

	const float e = (cos2_phi(wh) / (roughness * roughness) + sin2_phi(wh) / (roughness * roughness)) * t2t;
	return 1.0f / (PI * roughness * roughness * cos4_theta * (1.0f + e) * (1.0f + e));
}

__device__ float lambda(glm::vec3 w, float roughness)
{
	const float absTanTheta = abs(tan_theta(w));
	if (isinf(absTanTheta))
	{
		return 0.0f;
	}

	// Compute alpha for direction w
	const float alpha2_tan2_theta = (roughness * absTanTheta) * (roughness * absTanTheta);
	return (-1.0f + sqrt(1.0f + alpha2_tan2_theta)) / 2.0f;
}

__device__ float trowbridge_reitz_g(glm::vec3 wo, glm::vec3 wi, float roughness)
{
	return 1.0f / (1.0f + lambda(wo, roughness) + lambda(wi, roughness));
}

__device__ float trowbridge_reitz_pdf(glm::vec3 wo, glm::vec3 wh, float roughness)
{
	return trowbridge_reitz_d(wh, roughness) * abs_cos_theta(wh);
}

__device__ bool same_hemisphere(glm::vec3 w, glm::vec3 wp)
{
	return w.z * wp.z > 0.0f;
}

__device__ glm::vec3 sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness)
{
	const float phi = TWO_PI * xi[1];
	// We'll only handle isotropic microfacet materials
	const float tan_theta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
	const float cos_theta = 1.0f / sqrt(1.0f + tan_theta2);

	const float sin_theta =
	    sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));

	auto wh = glm::vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
	if (!same_hemisphere(wo, wh))
	{
		wh = -wh;
	}

	return wh;
}

__device__ glm::vec3 f_microfacet_refl(glm::vec3 albedo, glm::vec3 wo, glm::vec3 wi, float roughness)
{
	const float cos_theta_o = abs_cos_theta(wo);
	const float cos_theta_i = abs_cos_theta(wi);
	glm::vec3 wh = wi + wo;
	// Handle degenerate cases for microfacet reflection
	if (cos_theta_i == 0 || cos_theta_o == 0)
	{
		return glm::vec3(0.0f);
	}
	if (wh.x == 0 && wh.y == 0 && wh.z == 0)
	{
		return glm::vec3(0.0f);
	}
	wh = normalize(wh);
	// TODO: Handle different Fresnel coefficients
	const auto F = glm::vec3(1.0f);
	const float D = trowbridge_reitz_d(wh, roughness);
	const float G = trowbridge_reitz_g(wo, wi, roughness);
	return albedo * D * G * F / (4 * cos_theta_i * cos_theta_o);
}
