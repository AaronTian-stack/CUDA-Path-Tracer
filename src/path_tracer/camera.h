#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Terrible encapsulation of a camera
// The one from the base code doesn't even work correctly. There is warping at the poles
// and the translation only moves on the XZ plane which I don't think is intended.
class Camera
{
public:
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 look_at;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixel_length;
    float m_target_distance = 2.f;

    void update_vectors();
    void set_position(const glm::vec3& pos);
    void set_look_at(const glm::vec3& target);
    void rotate_around_target(float theta, float phi);
    void translate_local(float x, float y);
    void set_target_distance(float distance);
};
