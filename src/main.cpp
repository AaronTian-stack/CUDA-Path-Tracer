#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "application.h"
#include "tiny_gltf.h"
#include "path_tracer/path_tracer.h"
#include <argparse/argparse.hpp>

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("CUDA Path Tracer");

    program.add_argument("scene_file")
        .help("Path to the scene JSON file");

    program.add_argument("-bs2d", "--block-size-2d")
        .help("Block size for 2D kernels")
        .default_value(16)
		.nargs(1)
        .scan<'i', int>();

    program.add_argument("-bs1d", "--block-size-1d")
        .help("Block size for 1D kernels")
        .default_value(128)
		.nargs(1)
        .scan<'i', int>();

    program.add_argument("-ds", "--disable-save")
        .help("Disable saving the image to file")
        .default_value(false)
        .implicit_value(true);

    try 
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) 
    {
        printf("%s\n", err.what());
        printf("%s\n", program.usage().c_str());
        return 1;
    }

    const auto scene_file = program.get<std::string>("scene_file");
    const auto block_size_2d = program.get<int>("--block-size-2d");
    const auto block_size_1d = program.get<int>("--block-size-1d");
    const auto disable_save = program.get<bool>("--disable-save");

    PathTracer app;
    app.set_block_sizes(block_size_2d, block_size_1d, disable_save);
    if (!app.init_scene(scene_file.c_str()))
    {
        fprintf(stderr, "Failed to load scene. Make sure your working directory matches the scene file.\n");
        return EXIT_FAILURE;
    }
    app.run(false);

    return EXIT_SUCCESS;
}
