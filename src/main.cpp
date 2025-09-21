#include "cuda_test.h"
#include <glm/glm.hpp>

#define VOLK_IMPLEMENTATION
#include <volk.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

int main()
{
    VkResult result = volkInitialize();
    if (result != VK_SUCCESS) 
    {
        fprintf(stderr, "Failed to initialize volk: %d\n", result);
        return EXIT_FAILURE;
    }
    kernelWrapper();
    printf("host\n");
    glm::vec3 vec(1.0f, 2.0f, 3.0f);
    printf("GLM Vector: (%f, %f, %f)\n", vec.x, vec.y, vec.z);
    return EXIT_SUCCESS;
}
