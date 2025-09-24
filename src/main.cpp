#include "cuda_test.h"
#include <glm/glm.hpp>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "application.h"
#include "tiny_gltf.h"
#include "path_tracer/path_tracer.h"

int main()
{
    kernelWrapper();
    printf("host\n");
    glm::vec3 vec(1.0f, 2.0f, 3.0f);
    printf("GLM Vector: (%f, %f, %f)\n", vec.x, vec.y, vec.z);

    PathTracer app;
    app.run(true);

    return EXIT_SUCCESS;
}
