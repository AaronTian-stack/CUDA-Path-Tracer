cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

if(NOT WIN32)
    message(FATAL_ERROR "This project only supports Windows")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SOURCES
    src/main.cpp
    src/application.cpp
    src/path_tracer/cuda_pt.cu
    src/path_tracer/path_tracer.cpp
    src/path_tracer/buffers.cpp
    src/path_tracer/optix_denoiser.cpp
    src/vk/vk_window.cpp
    src/vk/vk_context.cpp
    src/vk/vk_cu_interop.cpp
)

set(HEADERS
    src/application.h
    src/path_tracer/cuda_pt.h
    src/path_tracer/path_tracer.h
    src/path_tracer/buffers.h
    src/path_tracer/optix_denoiser.h
    src/vk/vk_window.h
    src/vk/vk_context.h
    src/vk/vk_queue.h
    src/vk/vk_swapchain.h
    src/vk/vk_texture.h
    src/vk/vk_cu_interop.h
)

file(GLOB IMGUI_SOURCES external/imgui/*.cpp)

# Build ImGUI as a static library so that it doesn't show up in the path tracer project tree
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PRIVATE 
    external/imgui 
    external/volk-1.4.304 
    external/vulkan external/SDL-3.2.22/include)
target_compile_definitions(imgui PRIVATE 
    IMGUI_IMPL_VULKAN_USE_VOLK 
    IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
    VK_USE_PLATFORM_WIN32_KHR 
    VK_NO_PROTOTYPES 
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

# Similarly build vk-bootstrap as a static library
file(GLOB VK_BOOTSTRAP_SOURCES external/vk-bootstrap/*.cpp)
add_library(vk_bootstrap STATIC ${VK_BOOTSTRAP_SOURCES})
target_include_directories(vk_bootstrap PRIVATE external/vk-bootstrap external/vulkan)
target_compile_definitions(vk_bootstrap PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Don't include third party libraries in the VS project tree
set(ALL_PROJECT_FILES ${SOURCES} ${HEADERS})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_PROJECT_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_compile_definitions(${PROJECT_NAME} PRIVATE 
    IMGUI_IMPL_VULKAN_USE_VOLK 
    VK_USE_PLATFORM_WIN32_KHR 
    VULKAN_HPP_NO_EXCEPTIONS
    VULKAN_HPP_NO_CONSTRUCTORS
    VK_NO_PROTOTYPES
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

target_include_directories(${PROJECT_NAME} PRIVATE 
    external/imgui 
    external/volk-1.4.304 
    external/vulkan 
    external/glm
    external/SDL-3.2.22/include
    external/tinygltf-2.9.6
    external/vk-bootstrap
    external/Optix-9.0.0/include)

add_subdirectory(external/volk-1.4.304)

# TODO: link for Linux?
target_link_libraries(${PROJECT_NAME} PRIVATE 
    imgui
    vk_bootstrap
    ${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan/vulkan/bin/vulkan-1.lib 
    ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL-3.2.22/lib/SDL3.lib 
    volk)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL-3.2.22/lib/SDL3.dll
    $<TARGET_FILE_DIR:${PROJECT_NAME}>)

if(WIN32)
    # NVCC only works with MSVC on Windows so no need to check for it
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP>) # Enable multithread compilation
else()
    # TODO: Linux?
endif()