#pragma once

#include <string>
#include <vector>

#include "scene_structs.h"
#include "camera.h"

struct Scene
{
    bool load(const std::string& jsonName, SceneSettings* settings);

    Camera camera;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
};
