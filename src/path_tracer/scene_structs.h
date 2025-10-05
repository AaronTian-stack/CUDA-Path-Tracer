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

enum TonemapMode : uint8_t
{
	NONE,
	ACES,
	PBR_NEUTRAL,
};

struct PathTracerSettings
{
	int traced_depth = 0; // This is not a setting, only for display
	bool sort_rays = false;
	DisplayMode display_mode = PROGRESSIVE;
	TonemapMode tonemap_mode = ACES;
	int block_size_2d = 16;
	int block_size_1d = 128;
	bool disable_save = false;
};

struct SceneSettings
{
	int iterations = 5000;
	int trace_depth = 8;
    float exposure = 1.0f;
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
    struct
    {
        glm::vec4 factor{};
        int index = -1; // Texture index NOT image
    } base_color;
    struct
    {
        float metallic_factor = 0.0f;
        float roughness_factor = 0.0f;
        int index = -1;
    } metallic_roughness;
    struct
    {
        int index = -1;
        float scale = 1.0f;
    } normal;
    struct
    {
        int index = -1;
        float strength = 1.0f;
    } occlusion;
    struct
    {
        glm::vec3 factor{};
        int index = -1;
    } emissive;
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
	glm::vec2 uv;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
	float t;
	glm::vec3 surface_normal;
	int material_id;
	glm::vec2 uv;
	glm::vec4 tangent;
};
