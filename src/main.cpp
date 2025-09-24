#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "application.h"
#include "tiny_gltf.h"
#include "path_tracer/path_tracer.h"

int main()
{
    PathTracer app;
    app.run(true);

    return EXIT_SUCCESS;
}
