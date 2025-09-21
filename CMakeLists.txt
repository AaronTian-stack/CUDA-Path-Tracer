cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SOURCES
    src/main.cpp
    src/cuda_test.cu
)

set(HEADERS
    src/cuda_test.h
)

file(GLOB IMGUI_SOURCES external/imgui/*.cpp)

# Build ImGUI as a static library so that it doesn't show up in the path tracer project tree
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PRIVATE external/imgui external/volk-1.4.304 external/vulkan)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK VK_USE_PLATFORM_WIN32_KHR)

# Similarly build vk-bootstrap as a static library
file(GLOB VK_BOOTSTRAP_SOURCES external/vk-bootstrap/*.cpp)
add_library(vk_bootstrap STATIC ${VK_BOOTSTRAP_SOURCES})
target_include_directories(vk_bootstrap PRIVATE external/vk-bootstrap external/vulkan)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Don't include third party libraries in the VS project tree
set(ALL_PROJECT_FILES ${SOURCES} ${HEADERS})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_PROJECT_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_compile_definitions(${PROJECT_NAME} PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK VK_USE_PLATFORM_WIN32_KHR)

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

target_link_libraries(${PROJECT_NAME} PRIVATE 
    imgui
    vk_bootstrap
    ${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan/vulkan/bin/vulkan-1.lib 
    ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL-3.2.22/lib/SDL3.lib 
    volk_headers)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL-3.2.22/lib/SDL3.dll
    $<TARGET_FILE_DIR:${PROJECT_NAME}>)

if(WIN32)
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP>) # Enable multithread compilation
    endif()
endif()
