#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "application.h"
#include "tiny_gltf.h"
#include "path_tracer/path_tracer.h"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    PathTracer app;
    if (!app.init_scene(argv[1]))
    {
        fprintf(stderr, "Failed to load scene\n");
        return EXIT_FAILURE;
    }
    app.run(false);

    return EXIT_SUCCESS;
}
