#include "gltf_model.h"

#include <tiny_gltf.h>
#ifdef _WIN64
#include <windows.h>
#endif
#include <cuda_runtime.h>

void process_nodes(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	
}

void process_accessor_views(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	model->accessors.reserve(tiny_model.accessors.size());
    for (const auto& accessor : tiny_model.accessors)
    {
        model->accessors.emplace_back(pt::glTFModel::Accessor
            {
            .offset = accessor.byteOffset,
            .count = accessor.count,
            .type = accessor.type,
            .component_type = accessor.componentType,
            .buffer_view = accessor.bufferView,
        });
    }

    model->buffer_views.reserve(tiny_model.bufferViews.size());
    for (const auto& bufferView : tiny_model.bufferViews)
    {
        model->buffer_views.emplace_back(pt::glTFModel::BufferView
        {
            .offset = bufferView.byteOffset,
            .length = bufferView.byteLength,
            .stride = bufferView.byteStride,
            .buffer_index = bufferView.buffer,
        });
    }
}

void process_meshes(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	model->meshes.reserve(tiny_model.meshes.size());
	for (const auto& tiny_mesh : tiny_model.meshes)
	{
		std::vector<pt::glTFModel::Primitive> mesh;
		for (const auto& prim : tiny_mesh.primitives)
		{
			pt::glTFModel::Primitive p
			{
				.material_index = prim.material,
				.indices = prim.indices,
			};
			for (const auto& attr : prim.attributes)
			{
				p.attributes.emplace_back(attr.second);
			}
			mesh.push_back(std::move(p));
		}
		model->meshes.push_back(mesh);
	}
}

void process_materials(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	std::vector<pt::glTFModel::Material> materials;
	materials.reserve(tiny_model.materials.size());
	for (const auto& tiny_mat : tiny_model.materials)
	{
		const auto& tiny_pbr = tiny_mat.pbrMetallicRoughness;
		const auto& tiny_normal = tiny_mat.normalTexture;
		const auto& tiny_occlusion = tiny_mat.occlusionTexture;
		const auto& tiny_emissive = tiny_mat.emissiveTexture;
		materials.emplace_back(pt::glTFModel::Material{
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
				.texture_coordinate_set = tiny_pbr.baseColorTexture.texCoord,
			},
			.metallic_roughness =
			{
				.metallic_factor = static_cast<float>(tiny_pbr.metallicFactor),
				.roughness_factor = static_cast<float>(tiny_pbr.roughnessFactor),
				.index = tiny_pbr.metallicRoughnessTexture.index,
				.texture_coordinate_set = tiny_pbr.metallicRoughnessTexture.texCoord,
			},
			.normal =
			{
				.index = tiny_normal.index,
				.texture_coordinate_set = tiny_normal.texCoord,
				.scale = static_cast<float>(tiny_normal.scale),
			},
			.occlusion =
			{
				.index = tiny_occlusion.index,
				.texture_coordinate_set = tiny_occlusion.texCoord,
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
				.texture_coordinate_set = tiny_emissive.texCoord
			},
		});
	}
	cudaMalloc(reinterpret_cast<void**>(&model->cu_materials), sizeof(pt::glTFModel::Material) * materials.size());
	cudaMemcpy(model->cu_materials, materials.data(), sizeof(pt::glTFModel::Material) * materials.size(), cudaMemcpyHostToDevice);
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
        desc.sRGB = 0;
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
	model->cu_buffers.reserve(tiny_model.buffers.size());
	for (const auto& buffer : tiny_model.buffers)
	{
		void* cu_buf = nullptr;
		cudaMalloc(&cu_buf, buffer.data.size());
		cudaMemcpy(cu_buf, buffer.data.data(), buffer.data.size(), cudaMemcpyHostToDevice);
		model->cu_buffers.push_back(cu_buf);
	}
}

void process_textures(const tinygltf::Model& tiny_model, pt::glTFModel* model)
{
	model->cu_images.reserve(tiny_model.images.size());
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
		model->cu_images.push_back(cuArray);
	}
}

bool pt::glTFModel::load(const std::string& path)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    if (!warn.empty()) 
	{
		return false;
    }
    if (!err.empty()) 
	{
        return false;
    }
    if (!ret)
    {
		return false;
    }

    process_nodes(model, this);
    process_accessor_views(model, this);
	process_meshes(model, this);
	process_materials(model, this);
	auto sampler_desc = process_samplers(model, this);
	process_buffers(model, this);
	process_textures(model, this);

	// Just create textures directly since we combine image and sampler info
	cu_texture_sampler.reserve(model.textures.size());
	for (const auto& texture : model.textures)
	{
		cudaResourceDesc res_desc = {};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = this->cu_images[texture.source];
		cudaTextureObject_t tex_obj;
		cudaCreateTextureObject(&tex_obj, &res_desc, &sampler_desc[texture.sampler], nullptr);
		cu_texture_sampler.push_back(tex_obj);
	}

    return true;
}

pt::glTFModel::~glTFModel()
{
	for (void* buf : cu_buffers)
	{
		cudaFree(buf);
	}
	for (const auto& image : cu_images)
	{
		cudaFreeArray(image);
	}
}
