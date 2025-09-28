#pragma once
#include <string>
#include <cuda_runtime_api.h>
#include <vector>
#include <glm/glm.hpp>

#include "camera.h"
#include "scene_structs.h"

namespace pt
{
	// Partially adapted from my own open source project QhenkiX
	// Self plug: https://github.com/AaronTian-stack/QhenkiX
	struct glTFModel
	{
		bool load(const std::string& path, Camera* camera);

		struct Accessor
		{
			size_t offset = 0;
			size_t count = 0;
			int type = -1; // scalar, vector...
			int component_type = -1; // int, byte, short, float...
			int buffer_view = -1;
		};
		Accessor* d_accessors = nullptr; // CUDA device pointer to array

		// Pass data view info to CUDA kernel
		struct BufferView
		{
			size_t offset = 0;
			size_t length = 0;
			size_t stride = 0;
			int buffer_index = -1;
		};
		BufferView* d_buffer_views = nullptr;  
		void** d_buffers = nullptr; // Array of CUDA device pointers to actual data

		Material* d_materials = nullptr; // CUDA device pointer to array

		std::vector<cudaArray_t> d_images;
		cudaTextureObject_t* d_textures = nullptr;

		struct AABB
		{
			glm::vec3 min;
			glm::vec3 max;
		};

		struct Primitive
		{
			int material_index = -1; // material index
			int indices = -1; // accessor index
			int position_accessor = -1;
			int normal_accessor = -1;
			int texcoord_accessor = -1;
			int tangent_accessor = -1;
			AABB aabb;
		};
		Primitive* d_primitives = nullptr;

		struct HostNode
		{
			glm::mat4 local_transform;
			glm::mat4 global_transform;
			std::vector<int> children_indices;
			int parent_index = -1;
			int mesh_index = -1;
		};
		struct DeviceNode
		{
			glm::mat4 global_transform;
			int mesh_index = -1;
			int num_children = 0;
			int* children_indices = nullptr;
		};
		DeviceNode* d_nodes = nullptr;

		glm::mat4 camera_global_transform;
		int camera_index = -1;

		int root_node = -1;
		size_t num_buffers = 0;
		size_t num_textures = 0;
		size_t num_primitives = 0;
		size_t num_nodes = 0;
		size_t num_materials = 0;

		~glTFModel();
	};
}
