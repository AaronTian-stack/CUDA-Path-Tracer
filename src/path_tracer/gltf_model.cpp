#include "gltf_model.h"

#include <tiny_gltf.h>
#ifdef _WIN64
#define NOMINMAX
#include <windows.h>
#endif
#include <cuda_runtime.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

void process_nodes(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	std::vector<pt::glTFModel::HostNode> nodes;
	nodes.reserve(tiny_model.nodes.size());
	for (size_t i = 0; i < tiny_model.nodes.size(); ++i)
	{
		const auto& tiny_node = tiny_model.nodes[i];
		pt::glTFModel::HostNode node;
		node.parent_index = -1; // set later
		node.mesh_index = tiny_node.mesh;
		node.children_indices = tiny_node.children;

		// Compute local transform
		if (!tiny_node.matrix.empty())
		{
			for (int j = 0; j < 16; ++j)
			{
				node.local_transform[j / 4][j % 4] = tiny_node.matrix[j];
			}
		}
		else
		{
			glm::vec3 translation_vec = (tiny_node.translation.size() >= 3)
			    ? glm::vec3(tiny_node.translation[0], tiny_node.translation[1], tiny_node.translation[2])
			    : glm::vec3(0.0f);

			glm::quat rotation_quat = (tiny_node.rotation.size() >= 4)
			    ? glm::quat(tiny_node.rotation[3], tiny_node.rotation[0], tiny_node.rotation[1], tiny_node.rotation[2])
			    : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

			glm::vec3 scale_vec = (tiny_node.scale.size() >= 3)
			   ? glm::vec3(tiny_node.scale[0], tiny_node.scale[1], tiny_node.scale[2])
			   : glm::vec3(1.0f);

			glm::mat4 translation = glm::translate(glm::mat4(1.0f), translation_vec);
			glm::mat4 rotation = glm::mat4_cast(rotation_quat);
			glm::mat4 scale = glm::scale(glm::mat4(1.0f), scale_vec);
			node.local_transform = translation * rotation * scale;
		}

		nodes.push_back(node);
	}

	// Set parents and compute global transforms
	std::function<void(int, const glm::mat4&)> traverse = [&](int node_index, const glm::mat4& parent_transform)
	{
		nodes[node_index].global_transform = parent_transform * nodes[node_index].local_transform;
		if (tiny_model.nodes[node_index].camera != -1)
		{
			model->camera_global_transform = nodes[node_index].global_transform;
			model->camera_index = tiny_model.nodes[node_index].camera;
		}
		for (int child : nodes[node_index].children_indices)
		{
			traverse(child, nodes[node_index].global_transform);
		}
	};

	for (int scene_index : tiny_model.scenes[tiny_model.defaultScene].nodes)
	{
		traverse(scene_index, glm::mat4(1.0f));
	}

	// Create device nodes
	std::vector<pt::glTFModel::DeviceNode> device_nodes;
	device_nodes.reserve(nodes.size());
	for (const auto& host_node : nodes) 
	{
		pt::glTFModel::DeviceNode device_node;
		device_node.mesh_index = host_node.mesh_index;
		device_node.global_transform = host_node.global_transform;
		device_node.num_children = host_node.children_indices.size();
		if (device_node.num_children > 0)
		{
			cudaMalloc(&device_node.children_indices, sizeof(int) * device_node.num_children);
			cudaMemcpy(device_node.children_indices, host_node.children_indices.data(), sizeof(int) * device_node.num_children, cudaMemcpyHostToDevice);
		}
		else
		{
			device_node.children_indices = nullptr;
		}
		device_nodes.push_back(device_node);
	}
	cudaMalloc(reinterpret_cast<void**>(&model->d_nodes), sizeof(pt::glTFModel::DeviceNode) * device_nodes.size());
	cudaMemcpy(model->d_nodes, device_nodes.data(), sizeof(pt::glTFModel::DeviceNode) * device_nodes.size(), cudaMemcpyHostToDevice);
	model->num_nodes = device_nodes.size();
}

void process_accessor_views(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	std::vector<pt::glTFModel::Accessor> accessors;
	accessors.reserve(tiny_model.accessors.size());
    for (const auto& accessor : tiny_model.accessors)
    {
        accessors.emplace_back(pt::glTFModel::Accessor
            {
            .offset = accessor.byteOffset,
            .count = accessor.count,
            .type = accessor.type,
            .component_type = accessor.componentType,
            .buffer_view = accessor.bufferView,
        });
    }
	cudaMalloc(reinterpret_cast<void**>(&model->d_accessors), sizeof(pt::glTFModel::Accessor) * accessors.size());
	cudaMemcpy(model->d_accessors, accessors.data(), sizeof(pt::glTFModel::Accessor) * accessors.size(), cudaMemcpyHostToDevice);

	std::vector<pt::glTFModel::BufferView> buffer_views;
	buffer_views.reserve(tiny_model.bufferViews.size());
    for (const auto& bufferView : tiny_model.bufferViews)
    {
		buffer_views.emplace_back(pt::glTFModel::BufferView
        {
            .offset = bufferView.byteOffset,
            .length = bufferView.byteLength,
            .stride = bufferView.byteStride,
            .buffer_index = bufferView.buffer,
        });
    }
	cudaMalloc(reinterpret_cast<void**>(&model->d_buffer_views), sizeof(pt::glTFModel::BufferView) * buffer_views.size());
	cudaMemcpy(model->d_buffer_views, buffer_views.data(), sizeof(pt::glTFModel::BufferView) * buffer_views.size(), cudaMemcpyHostToDevice);
}

bool process_meshes(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	cudaMalloc(reinterpret_cast<void**>(&model->d_primitives), sizeof(pt::glTFModel::Primitive) * tiny_model.meshes.size());
	for (int i = 0; i < tiny_model.meshes.size(); i++)
	{
		const auto& tiny_mesh = tiny_model.meshes[i];
		if (tiny_mesh.primitives.size() > 1)
		{
			return false;
		}
		const auto& prim = tiny_mesh.primitives[0];
		pt::glTFModel::Primitive p
		{
			.material_index = prim.material,
			.indices = prim.indices,
		};
		if (prim.attributes.count("POSITION")) p.position_accessor = prim.attributes.at("POSITION");
		if (prim.attributes.count("NORMAL")) p.normal_accessor = prim.attributes.at("NORMAL");
		if (prim.attributes.count("TEXCOORD_0")) p.texcoord_accessor = prim.attributes.at("TEXCOORD_0");
		if (prim.attributes.count("TANGENT")) p.tangent_accessor = prim.attributes.at("TANGENT");

		// Compute AABB
		if (p.position_accessor != -1)
		{
			const auto& pos_acc = tiny_model.accessors[p.position_accessor];
			const auto& pos_bv = tiny_model.bufferViews[pos_acc.bufferView];
			const auto& buffer = tiny_model.buffers[pos_bv.buffer];
			size_t pos_offset = pos_acc.byteOffset + pos_bv.byteOffset;
			size_t pos_stride = pos_bv.byteStride;
			if (pos_stride == 0) pos_stride = 3 * sizeof(float);
			size_t num_vertices = pos_acc.count;

			glm::vec3 min_pos(FLT_MAX);
			glm::vec3 max_pos(-FLT_MAX);

			for (size_t j = 0; j < num_vertices; ++j)
			{
				const float* pos = reinterpret_cast<const float*>(&buffer.data[pos_offset + j * pos_stride]);
				glm::vec3 vertex(pos[0], pos[1], pos[2]);
				min_pos = glm::min(min_pos, vertex);
				max_pos = glm::max(max_pos, vertex);
			}
			p.aabb.min = min_pos - glm::vec3(1e-5f);
			p.aabb.max = max_pos + glm::vec3(1e-5f);
		}
		else
		{
			p.aabb.min = glm::vec3(0.0f);
			p.aabb.max = glm::vec3(0.0f);
		}

		cudaMemcpy(model->d_primitives + i, &p, sizeof(pt::glTFModel::Primitive), cudaMemcpyHostToDevice);
	}
	return true;
}

void process_materials(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	std::vector<Material> materials;
	materials.reserve(tiny_model.materials.size());
	for (const auto& tiny_mat : tiny_model.materials)
	{
		const auto& tiny_pbr = tiny_mat.pbrMetallicRoughness;
		const auto& tiny_normal = tiny_mat.normalTexture;
		const auto& tiny_occlusion = tiny_mat.occlusionTexture;
		const auto& tiny_emissive = tiny_mat.emissiveTexture;
		materials.emplace_back(Material{
			.base_color =
			{
				.factor = 
				{
					static_cast<float>(tiny_pbr.baseColorFactor[0]),
					static_cast<float>(tiny_pbr.baseColorFactor[1]),
					static_cast<float>(tiny_pbr.baseColorFactor[2]),
					static_cast<float>(tiny_pbr.baseColorFactor[3]),
				},
				.index = tiny_pbr.baseColorTexture.index,
			},
			.metallic_roughness =
			{
				.metallic_factor = static_cast<float>(tiny_pbr.metallicFactor),
				.roughness_factor = static_cast<float>(tiny_pbr.roughnessFactor),
				.index = tiny_pbr.metallicRoughnessTexture.index,
			},
			.normal =
			{
				.index = tiny_normal.index,
				.scale = static_cast<float>(tiny_normal.scale),
			},
			.occlusion =
			{
				.index = tiny_occlusion.index,
				.strength = static_cast<float>(tiny_occlusion.strength),
			},
			.emissive = 
			{
				.factor =
				{
					static_cast<float>(tiny_mat.emissiveFactor[0]),
					static_cast<float>(tiny_mat.emissiveFactor[1]),
					static_cast<float>(tiny_mat.emissiveFactor[2]),
				},
				.index = tiny_emissive.index,
			},
		});
	}
	cudaMalloc(reinterpret_cast<void**>(&model->d_materials), sizeof(Material) * materials.size());
	cudaMemcpy(model->d_materials, materials.data(), sizeof(Material) * materials.size(), cudaMemcpyHostToDevice);
	model->num_materials = materials.size();
}

std::vector<cudaTextureDesc> process_samplers(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	std::vector<cudaTextureDesc> samplers;
    samplers.reserve(tiny_model.samplers.size());
    for (const auto& tiny_sampler : tiny_model.samplers)
    {
	    samplers.emplace_back();
        auto& desc = samplers.back();
        switch (tiny_sampler.minFilter)
        {
	        default:
#ifdef _WIN64
	            OutputDebugStringA("WARNING: Using default filter\n");
#endif
            case TINYGLTF_TEXTURE_FILTER_NEAREST:
                desc.filterMode = cudaFilterModePoint;
                desc.mipmapFilterMode = cudaFilterModePoint;
                break;
            case TINYGLTF_TEXTURE_FILTER_LINEAR:
                desc.filterMode = cudaFilterModeLinear;
                desc.mipmapFilterMode = cudaFilterModeLinear;
                break;
        }
        switch (tiny_sampler.wrapS)
        {
			default:
#ifdef _WIN64
				OutputDebugStringA("WARNING: Using default wrap U filter\n");
#endif
            case TINYGLTF_TEXTURE_WRAP_REPEAT:
                desc.addressMode[0] = cudaAddressModeWrap;
                break;
            case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
                desc.addressMode[0] = cudaAddressModeClamp;
                break;
            case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
                desc.addressMode[0] = cudaAddressModeMirror;
                break;
        }
		switch (tiny_sampler.wrapT)
		{
		default:
#ifdef _WIN64
			OutputDebugStringA("WARNING: Using default wrap T filter\n");
#endif
		case TINYGLTF_TEXTURE_WRAP_REPEAT:
			desc.addressMode[1] = cudaAddressModeWrap;
			break;
		case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
			desc.addressMode[1] = cudaAddressModeClamp;
			break;
		case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
			desc.addressMode[1] = cudaAddressModeMirror;
			break;
		}
        desc.addressMode[2] = cudaAddressModeWrap;
        desc.readMode = cudaReadModeNormalizedFloat;
        desc.sRGB = 1;
        desc.normalizedCoords = 1;
        desc.maxAnisotropy = 1;
        desc.mipmapLevelBias = 0.0f;
        desc.minMipmapLevelClamp = 0.0f;
        desc.maxMipmapLevelClamp = 1000.0f;
    }
	return samplers;
}

void process_buffers(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	model->num_buffers = tiny_model.buffers.size();
	cudaMalloc(reinterpret_cast<void**>(&model->d_buffers), sizeof(void*) * model->num_buffers);
	for (size_t i = 0; i < tiny_model.buffers.size(); ++i)
	{
		const auto& buffer = tiny_model.buffers[i];
		void* cu_buf = nullptr;
		cudaMalloc(&cu_buf, buffer.data.size());
		cudaMemcpy(cu_buf, buffer.data.data(), buffer.data.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(model->d_buffers + i, &cu_buf, sizeof(void*), cudaMemcpyHostToDevice);
	}
}

void process_textures(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	model->d_images.reserve(tiny_model.images.size());
	for (const auto& tiny_image : tiny_model.images)
	{
		cudaChannelFormatDesc channelDesc;
		if (tiny_image.component == 4)
		{
			channelDesc = cudaCreateChannelDesc<uchar4>();
		}
		else if (tiny_image.component == 3)
		{
			channelDesc = cudaCreateChannelDesc<uchar3>();
		}
		else if (tiny_image.component == 2)
		{
			channelDesc = cudaCreateChannelDesc<uchar2>();
		}
		else
		{
			channelDesc = cudaCreateChannelDesc<unsigned char>();
		}
		cudaArray_t cuArray;
		cudaMallocArray(&cuArray, &channelDesc, tiny_image.width, tiny_image.height);
		size_t pitch = tiny_image.width * tiny_image.component * sizeof(unsigned char);
		cudaMemcpy2DToArray(cuArray, 0, 0, tiny_image.image.data(), pitch, pitch, tiny_image.height, cudaMemcpyHostToDevice);
		model->d_images.push_back(cuArray);
	}
}

bool pt::glTFModel::load(const std::string& path, Camera* camera)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ret;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb")
    {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    }
    else if (path.size() >= 5 && path.substr(path.size() - 5) == ".gltf")
    {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }
    else
    {
        err = "Unsupported file extension. Expected .glb or .gltf";
        ret = false;
    }
    if (!warn.empty())
	{
		printf("%s\n", warn.c_str());
		return false;
	}
	if (!err.empty() || !ret)
	{
		printf("%s\n", err.c_str());
		return false;
	}

    process_nodes(model, this);
    process_accessor_views(model, this);
	if (!process_meshes(model, this)) return false;

	process_materials(model, this);
	const auto sampler_desc = process_samplers(model, this);
	process_buffers(model, this);
	process_textures(model, this);

	this->num_textures = model.textures.size();
	// Just create textures directly since we combine image and sampler info
	cudaMalloc(reinterpret_cast<void**>(&this->d_textures), sizeof(cudaTextureObject_t) * this->num_textures);
	for (int i = 0; i < model.textures.size(); i++)
	{
		const auto& texture = model.textures[i];
		cudaResourceDesc res_desc = {};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = this->d_images[texture.source];

		cudaTextureObject_t tex_obj;
		if (texture.sampler >= 0)
		{
			cudaCreateTextureObject(&tex_obj, &res_desc, &sampler_desc[texture.sampler], nullptr);
		}
		else
		{
			cudaTextureDesc tex_desc = 
			{
				.addressMode = {cudaAddressModeWrap, cudaAddressModeWrap, cudaAddressModeWrap},
				.filterMode = cudaFilterModeLinear,
				.readMode = cudaReadModeNormalizedFloat,
				.sRGB = 1,
				.borderColor = {0.0f, 0.0f, 0.0f, 0.0f},
				.normalizedCoords = 1,
				.maxAnisotropy = 1,
				.mipmapFilterMode = cudaFilterModeLinear,
				.mipmapLevelBias = 0.0f,
				.minMipmapLevelClamp = 0.0f,
				.maxMipmapLevelClamp = 1000.0f,
				.disableTrilinearOptimization = 0,
				.seamlessCubemap = 0,
			};
			cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
		}
		cudaMemcpy(this->d_textures + i, &tex_obj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
	}

	// Set camera from glTF if available
	if (camera && camera_index != -1)
	{
		const auto& tiny_camera = model.cameras[camera_index];
		if (tiny_camera.type == "perspective")
		{
			const auto& pers = tiny_camera.perspective;
			float half_yfov = static_cast<float>(pers.yfov) / 2.0f;
			float aspect;
			if (pers.aspectRatio > 0.0)
			{
				aspect = static_cast<float>(pers.aspectRatio);
				// Adjust horizontal resolution to match glTF camera's aspect ratio
				camera->resolution.x = static_cast<int>(camera->resolution.y * pers.aspectRatio);
			}
			else
			{
				aspect = static_cast<float>(camera->resolution.x) / static_cast<float>(camera->resolution.y);
			}
			float half_xfov = atan(tan(half_yfov) * aspect);
			camera->fov.x = glm::degrees(half_xfov);
			camera->fov.y = glm::degrees(half_yfov);
			// Update pixel_length based on fov and resolution
			float yscaled = tan(half_yfov);
			float xscaled = yscaled * aspect;
			camera->pixel_length = glm::vec2(2 * xscaled / static_cast<float>(camera->resolution.x),
				2 * yscaled / static_cast<float>(camera->resolution.y));
			glm::vec3 position = glm::vec3(camera_global_transform[3]);
			glm::vec3 view_dir = glm::normalize(glm::vec3(camera_global_transform * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
			glm::vec3 look_at = position + view_dir;
			glm::vec3 up = glm::normalize(glm::vec3(camera_global_transform * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
			camera->position = position;
			camera->look_at = look_at;
			camera->up = up;
			camera->update_vectors();
		}
	}

    return true;
}

pt::glTFModel::~glTFModel()
{
	cudaFree(d_accessors);
	cudaFree(d_buffer_views);
	for (int i = 0; i < num_buffers; i++)
	{
		void* buf;
		cudaMemcpy(&buf, d_buffers + i, sizeof(void*), cudaMemcpyDeviceToHost);
		cudaFree(buf);
	}
	cudaFree(d_buffers);
	for (const auto& image : d_images)
	{
		cudaFreeArray(image);
	}
	for (int i = 0; i < num_textures; i++)
	{
		cudaTextureObject_t tex;
		cudaMemcpy(&tex, d_textures + i, sizeof(cudaTextureObject_t), cudaMemcpyDeviceToHost);
		cudaDestroyTextureObject(tex);
	}
	cudaFree(d_textures);
	cudaFree(d_primitives);
	cudaFree(d_materials);
	if (d_nodes)
	{
		// Copy all device nodes to host to free their children
		std::vector<DeviceNode> host_nodes(num_nodes);
		cudaMemcpy(host_nodes.data(), d_nodes, sizeof(DeviceNode) * num_nodes, cudaMemcpyDeviceToHost);
		for (const auto& node : host_nodes)
		{
			if (node.children_indices)
			{
				cudaFree(node.children_indices);
			}
		}
		cudaFree(d_nodes);
	}
}
