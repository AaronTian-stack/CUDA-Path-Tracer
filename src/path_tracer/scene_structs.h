#pragma once
#include <glm/glm.hpp>
#include <string>

enum DisplayMode : uint8_t
{
	PROGRESSIVE,
	ALBEDO,
	NORMAL,
	DENOISED,
};

struct PathTracerSettings
{
	int traced_depth = 0; // This is not a setting, only for display
	bool sort_rays = false;
	DisplayMode display_mode = PROGRESSIVE;
};

struct SceneSettings
{
	int iterations = 5000;
	int trace_depth = 8;
	std::string output_name;
};

enum GeomType : uint8_t
{
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    GeomType type;
    int material_id;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    glm::vec4 albedo;
    glm::vec3 emissive;
    float metallic;
    float roughness;
    // TODO: extension materials like transmission?
};

struct PathSegments
{
    glm::vec3* origins;
    glm::vec3* directions;
    glm::vec3* colors;
    int* pixel_indices;
    int* remaining_bounces;
};

struct IntersectionData
{
	glm::vec3 position;
	glm::vec3 normal;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
	float t;
	glm::vec3 surface_normal;
	int material_id;
};
