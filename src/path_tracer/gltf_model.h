#pragma once
#include <string>
#include <cuda_runtime_api.h>
#include <vector>
#include <glm/glm.hpp>

namespace pt
{
	// Partially adapted from my own open source project QhenkiX
	// Self plug: https://github.com/AaronTian-stack/QhenkiX
	struct glTFModel
	{
		bool load(const std::string& path);

		struct Accessor
		{
			size_t offset = 0;
			size_t count = 0;
			int type = -1; // scalar, vector...
			int component_type = -1; // int, byte, short, float...
			int buffer_view = -1;
		};
		std::vector<Accessor> accessors; // CPU

		// Pass data view info to CUDA kernel
		struct BufferView
		{
			size_t offset = 0;
			size_t length = 0;
			size_t stride = 0;
			int buffer_index = -1;
		};
		std::vector<BufferView> buffer_views; // CPU
		std::vector<void*> cu_buffers; // Vector of CUDA device pointers to actual data

		struct Material
		{
			struct
			{
				glm::vec4 factor;
				int index = -1; // Texture index NOT image
				int texture_coordinate_set = 0;
			} base_color;
			struct
			{
				float metallic_factor = 0.f;
				float roughness_factor = 0.f;
				int index = -1;
				int texture_coordinate_set = 0;
			} metallic_roughness;
			struct
			{
				int index = -1;
				int texture_coordinate_set = 0;
				float scale = 1.f;
			} normal;
			struct
			{
				int index = -1;
				int texture_coordinate_set = 0;
				float strength = 1.f;
			} occlusion;
			struct
			{
				glm::vec3 factor;
				int index = -1;
				int texture_coordinate_set = 0;
			} emissive;
		};
		Material* cu_materials = nullptr; // CUDA device pointer to array

		std::vector<cudaArray_t> cu_images;
		std::vector<cudaTextureObject_t> cu_texture_sampler;

		struct Primitive
		{
			int material_index = -1; // material index
			int indices = -1; // accessor index
			std::vector<int> attributes; // Accessor index
		};
		std::vector<std::vector<Primitive>> meshes;

		struct Node
		{
			int parent_index = -1;
			int mesh_index = -1;
			glm::mat4 local_transform;
			glm::mat4 global_transform;
			std::vector<int> children_indices;
		};
		std::vector<Node> nodes;

		int root_node = -1;

		~glTFModel();
	};
}
