#include "scene.h"

#include <fstream>
#include <json.hpp>

#include "util.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/euler_angles.hpp>

using json = nlohmann::json;

bool Scene::load(const std::string& file_name, SceneSettings* settings)
{
    const auto ext = file_name.substr(file_name.find_last_of('.'));
    if (ext != ".json")
    {
        return false;
    }

    std::ifstream f(file_name);
    nlohmann::json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> mat_name_to_id;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material new_material{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            new_material.albedo = glm::vec4(col[0], col[1], col[2], 1.f);
            new_material.roughness = 1.0f;
            new_material.metallic = 0.0f;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            new_material.emissive = glm::vec3(col[0], col[1], col[2]) * static_cast<float>(p["EMITTANCE"]);
            new_material.albedo = glm::vec4(col[0], col[1], col[2], 1.0f); // Set albedo for emissive too
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            new_material.albedo = glm::vec4(col[0], col[1], col[2], 1.0f);
            new_material.roughness = p.contains("ROUGHNESS") ? static_cast<float>(p["ROUGHNESS"]) : 0.f;
            new_material.metallic = p.contains("METALLIC") ? static_cast<float>(p["METALLIC"]) : 0.f;
        }
        mat_name_to_id[name] = materials.size();
        materials.emplace_back(new_material);
    }
    const auto& objects_data = data["Objects"];
    for (const auto& p : objects_data)
    {
        const auto& type = p["TYPE"];
        Geom new_geom;
        if (type == "cube")
        {
            new_geom.type = CUBE;
        }
        else
        {
            new_geom.type = SPHERE;
        }
        new_geom.material_id = mat_name_to_id[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        new_geom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        new_geom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        new_geom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        {
            auto translate_mat = glm::translate(glm::mat4(1.f), new_geom.translation);
			auto rotate_mat = glm::eulerAngleXYZ(glm::radians(new_geom.rotation.x), glm::radians(new_geom.rotation.y), glm::radians(new_geom.rotation.z));
            auto scale_mat = glm::scale(glm::mat4(1.f), new_geom.scale);
			new_geom.transform = translate_mat * rotate_mat * scale_mat;
        }
        new_geom.inverseTransform = glm::inverse(new_geom.transform);
        new_geom.invTranspose = glm::inverseTranspose(new_geom.transform);

        geoms.push_back(new_geom);
    }
    const auto& camera_data = data["Camera"];
    camera.resolution.x = camera_data["RES"][0];
    camera.resolution.y = camera_data["RES"][1];
    float fovy = camera_data["FOVY"];
    if (settings)
    {
        settings->iterations = camera_data["ITERATIONS"];
        settings->trace_depth = camera_data["DEPTH"];
        settings->output_name = camera_data["FILE"];
    }
    const auto& pos = camera_data["EYE"];
    const auto& lookat = camera_data["LOOKAT"];
    const auto& up = camera_data["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.look_at = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.pixel_length = glm::vec2(2 * xscaled / static_cast<float>(camera.resolution.x),
        2 * yscaled / static_cast<float>(camera.resolution.y));

	camera.update_vectors();

    return true;
}
