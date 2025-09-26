#include "camera.h"

void Camera::update_vectors()
{
    view = glm::normalize(look_at - position);
    right = glm::normalize(glm::cross(up, view));
    up = glm::normalize(-glm::cross(right, view));
}

void Camera::set_position(const glm::vec3& pos)
{
    position = pos;
    update_vectors();
}

void Camera::set_look_at(const glm::vec3& target)
{
    look_at = target;
    update_vectors();
}

void Camera::rotate_around_target(float theta, float phi)
{
    // Compute current angle with up vector
    glm::vec3 to_target = position - look_at;
    glm::vec3 dir = glm::normalize(to_target);
    float current_angle = glm::acos(glm::clamp(glm::dot(dir, glm::vec3(0.0f, 1.0f, 0.0f)), -1.0f, 1.0f));

    const float max_tolerance = 0.00174533f; // ~0.1 degrees

    // Clamp phi to prevent flipping over poles
    if (current_angle + phi < max_tolerance)
    {
        phi = max_tolerance - current_angle;
    }
    if (current_angle + phi > glm::pi<float>() - max_tolerance)
    {
        phi = glm::pi<float>() - current_angle - max_tolerance;
    }

    // Rotate around current right (local X) for phi
    glm::mat4 rot_phi = glm::rotate(glm::mat4(1.0f), phi, right);
    glm::vec3 new_pos = look_at + glm::vec3(rot_phi * glm::vec4(to_target, 1.0f));

    // Rotate around global Y for theta
    glm::mat4 rot_theta = glm::rotate(glm::mat4(1.0f), theta, glm::vec3(0.0f, 1.0f, 0.0f));
    position = look_at + glm::vec3(rot_theta * glm::vec4(new_pos - look_at, 1.0f));

    // Rotate the basis vectors
    up = glm::normalize(glm::vec3(rot_theta * glm::vec4(glm::vec3(rot_phi * glm::vec4(up, 0.0f)), 0.0f)));
    right = glm::normalize(glm::vec3(rot_theta * glm::vec4(glm::vec3(rot_phi * glm::vec4(right, 0.0f)), 0.0f)));
    view = glm::normalize(look_at - position);
}

void Camera::translate_local(float x, float y)
{
    glm::vec3 delta = right * x + up * y;
    position += delta;
    look_at += delta;
}

void Camera::set_target_distance(float distance)
{
    distance = glm::max(distance, 0.01f);
    m_target_distance = distance;
    glm::vec3 dir = glm::normalize(position - look_at);
    position = look_at + dir * distance;
    update_vectors();
}
