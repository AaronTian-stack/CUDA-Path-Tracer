#pragma once
#include <surface_types.h>

void test_set_image(cudaSurfaceObject_t surf_obj, size_t width, size_t height, float time, cudaExternalSemaphore_t sem);