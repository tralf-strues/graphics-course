#include "SSSRPass.hpp"

#include <etna/RenderTargetStates.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>

#include <stb_image.h>

auto binding_sampled(uint32_t binding, const etna::Image& image, const etna::Sampler& sampler)
{
  return etna::Binding(
    binding,
    image.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, {}));
}

auto binding_sampled_cube(uint32_t binding, const etna::Image& image, const etna::Sampler& sampler)
{
  return etna::Binding(
    binding,
    image.genBinding(
      sampler.get(),
      vk::ImageLayout::eShaderReadOnlyOptimal,
      etna::Image::ViewParams{
        0,
        vk::RemainingMipLevels,
        0,
        vk::RemainingArrayLayers,
        {},
        vk::ImageViewType::eCube,
      }));
}

auto binding_sampled_array(uint32_t binding, const etna::Image& image, const etna::Sampler& sampler)
{
  return etna::Binding(
    binding,
    image.genBinding(
      sampler.get(),
      vk::ImageLayout::eShaderReadOnlyOptimal,
      etna::Image::ViewParams{
        0,
        vk::RemainingMipLevels,
        0,
        vk::RemainingArrayLayers,
        {},
        vk::ImageViewType::e2DArray,
      }));
}

auto binding_write(uint32_t binding, const etna::Image& image)
{
  return etna::Binding(binding, image.genBinding(nullptr, vk::ImageLayout::eGeneral, {}));
}

void SSSRPass::loadShaders()
{
  etna::create_program("sssr_reflect", {DEMO_SHADERS_ROOT "sssr_reflect.comp.spv"});
  etna::create_program("sssr_ta", {DEMO_SHADERS_ROOT "sssr_ta.comp.spv"});
  etna::create_program("sssr_filter", {DEMO_SHADERS_ROOT "sssr_filter.comp.spv"});
}

void SSSRPass::allocateResources(glm::uvec2 resolution, vk::Format format)
{
  auto& ctx = etna::get_context();

  pointSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eClampToBorder,
    .name = "SSSRPass::pointSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
  });

  linearSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eClampToBorder,
    .name = "SSSRPass::linearSampler",
    .minLod = 0.0f,
    .maxLod = vk::LodClampNone,
  });

  linearSamplerRepeat = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "SSSRPass::linearSamplerRepeat",
    .minLod = 0.0f,
    .maxLod = vk::LodClampNone,
  });

  reflectTargetReflection = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "SSSRPass::reflectTargetReflection",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });

  reflectTargetReprojectionUV = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "SSSRPass::reflectTargetReprojectionUV",
    .format = vk::Format::eR16G16Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });

  glm::uvec2 avgReflectionResolution = resolution / 8U;
  avgReflection = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{avgReflectionResolution.x, avgReflectionResolution.y, 1},
    .name = "SSSRPass::avgReflection",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });

  taTarget = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "SSSRPass::taTarget",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });

  for (size_t i = 0; i < taVarianceAndNumSamples.size(); ++i)
  {
    taVarianceAndNumSamples[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "SSSRPass::taVarianceAndNumSamples[" + std::to_string(i) + "]",
      .format = vk::Format::eR16G16Sfloat,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eTransferDst,
    });
  }

  filterTarget = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "SSSRPass::filterTarget",
    .format = format,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
      vk::ImageUsageFlagBits::eTransferDst |
      vk::ImageUsageFlagBits::eTransferSrc, // TODO: Remove transfer src
  });
}

void SSSRPass::setupPipelines()
{
  reflectPipeline =
    etna::get_context().getPipelineManager().createComputePipeline("sssr_reflect", {});

  taPipeline = etna::get_context().getPipelineManager().createComputePipeline("sssr_ta", {});

  filterPipeline =
    etna::get_context().getPipelineManager().createComputePipeline("sssr_filter", {});

  readBlueNoise();
}

void SSSRPass::execute(
  vk::CommandBuffer cmds,
  const Params& params,
  etna::Buffer& prev_camera_buffer,
  etna::Buffer& curr_camera_buffer,
  etna::Image& hiz,
  etna::Image& prev_depth,
  Temporal<etna::Image>& gbuffer_norm,
  etna::Image& curr_motion_vectors,
  etna::Image& prev_color,
  etna::Image& gbuffer_metalness_roughness,
  const etna::Image& prefiltered_environment_map)
{
  etna::set_state(
    cmds,
    hiz.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmds,
    prev_depth.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eDepth);

  etna::set_state(
    cmds,
    gbuffer_norm.getCurrent().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmds,
    gbuffer_norm.getPrevious().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmds,
    curr_motion_vectors.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmds,
    prev_color.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmds,
    gbuffer_metalness_roughness.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  // Reflect
  {
    ETNA_PROFILE_GPU(cmds, SSSR_Reflect);

    etna::set_state(
      cmds,
      filterTarget.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      reflectTargetReflection.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      reflectTargetReprojectionUV.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      avgReflection.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::flush_barriers(cmds);

    cmds.bindPipeline(vk::PipelineBindPoint::eCompute, reflectPipeline.getVkPipeline());

    auto programInfo = etna::get_shader_program("sssr_reflect");
    auto descriptorSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(0),
      cmds,
      {
        etna::Binding(0, prev_camera_buffer.genBinding()),
        etna::Binding(1, curr_camera_buffer.genBinding()),
        binding_sampled(2, hiz, pointSampler),
        binding_sampled(3, prev_depth, pointSampler),
        binding_sampled(4, gbuffer_norm.getCurrent(), linearSampler),
        binding_sampled(5, curr_motion_vectors, linearSampler),
        binding_sampled(6, prev_color, linearSampler),
        binding_sampled(7, gbuffer_metalness_roughness, linearSampler),
        binding_sampled_cube(8, prefiltered_environment_map, linearSamplerRepeat),
        binding_sampled_array(9, blueNoise, pointSampler),
        binding_write(10, reflectTargetReflection),
        binding_write(11, reflectTargetReprojectionUV),
        binding_write(12, avgReflection),
      });

    cmds.bindPipeline(vk::PipelineBindPoint::eCompute, reflectPipeline.getVkPipeline());
    cmds.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      reflectPipeline.getVkPipelineLayout(),
      0,
      {descriptorSet.getVkSet()},
      {});

    cmds.pushConstants<Params>(
      programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {params});

    cmds.dispatch(
      (params.resolution.x + GROUP_SIZE - 1) / GROUP_SIZE,
      (params.resolution.y + GROUP_SIZE - 1) / GROUP_SIZE,
      1);
  }

  // Temporal accumulation
  {
    ETNA_PROFILE_GPU(cmds, SSSR_TA);

    etna::set_state(
      cmds,
      reflectTargetReflection.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      reflectTargetReprojectionUV.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      avgReflection.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      filterTarget.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      taVarianceAndNumSamples.getPrevious().get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      taTarget.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      taVarianceAndNumSamples.getCurrent().get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::flush_barriers(cmds);

    cmds.bindPipeline(vk::PipelineBindPoint::eCompute, taPipeline.getVkPipeline());

    auto programInfo = etna::get_shader_program("sssr_ta");
    auto descriptorSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(0),
      cmds,
      {
        etna::Binding(0, prev_camera_buffer.genBinding()),
        etna::Binding(1, curr_camera_buffer.genBinding()),
        binding_sampled(2, hiz, pointSampler),
        binding_sampled(3, prev_depth, pointSampler),
        binding_sampled(4, gbuffer_norm.getPrevious(), linearSampler),
        binding_sampled(5, gbuffer_norm.getCurrent(), pointSampler),
        binding_sampled(6, gbuffer_metalness_roughness, pointSampler),
        binding_sampled(7, filterTarget, linearSampler),
        binding_sampled(8, taVarianceAndNumSamples.getPrevious(), linearSampler),
        binding_sampled(9, reflectTargetReflection, pointSampler),
        binding_sampled(10, reflectTargetReprojectionUV, pointSampler),
        binding_sampled(11, avgReflection, linearSampler),
        binding_write(12, taTarget),
        binding_write(13, taVarianceAndNumSamples.getCurrent()),
      });

    cmds.bindPipeline(vk::PipelineBindPoint::eCompute, taPipeline.getVkPipeline());
    cmds.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      taPipeline.getVkPipelineLayout(),
      0,
      {descriptorSet.getVkSet()},
      {});

    cmds.pushConstants<Params>(
      programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {params});

    cmds.dispatch(
      (params.resolution.x + GROUP_SIZE - 1) / GROUP_SIZE,
      (params.resolution.y + GROUP_SIZE - 1) / GROUP_SIZE,
      1);
  }

  taVarianceAndNumSamples.proceed();

  // Filter
  {
    ETNA_PROFILE_GPU(cmds, SSSR_Filter);

    etna::set_state(
      cmds,
      taTarget.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      taVarianceAndNumSamples.getPrevious().get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      filterTarget.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmds,
      taVarianceAndNumSamples.getCurrent().get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::flush_barriers(cmds);

    cmds.bindPipeline(vk::PipelineBindPoint::eCompute, filterPipeline.getVkPipeline());

    auto programInfo = etna::get_shader_program("sssr_filter");
    auto descriptorSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(0),
      cmds,
      {
        etna::Binding(0, curr_camera_buffer.genBinding()),
        binding_sampled(1, hiz, pointSampler),
        binding_sampled(2, gbuffer_norm.getCurrent(), pointSampler),
        binding_sampled(3, gbuffer_metalness_roughness, pointSampler),
        binding_sampled(4, taTarget, pointSampler),
        binding_sampled(5, taVarianceAndNumSamples.getPrevious(), pointSampler),
        binding_sampled(6, avgReflection, linearSampler),
        binding_write(7, filterTarget),
        binding_write(8, taVarianceAndNumSamples.getCurrent()),
      });

    cmds.bindPipeline(vk::PipelineBindPoint::eCompute, filterPipeline.getVkPipeline());
    cmds.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      filterPipeline.getVkPipelineLayout(),
      0,
      {descriptorSet.getVkSet()},
      {});

    cmds.pushConstants<Params>(
      programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {params});

    cmds.dispatch(
      (params.resolution.x + GROUP_SIZE - 1) / GROUP_SIZE,
      (params.resolution.y + GROUP_SIZE - 1) / GROUP_SIZE,
      1);
  }
}

void SSSRPass::invalidate(vk::CommandBuffer cmds)
{
  etna::set_state(
    cmds,
    filterTarget.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferWrite,
    vk::ImageLayout::eTransferDstOptimal,
    vk::ImageAspectFlagBits::eColor);

  for (auto& image : taVarianceAndNumSamples)
  {
    etna::set_state(
      cmds,
      image.get(),
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eTransferDstOptimal,
      vk::ImageAspectFlagBits::eColor);
  }

  etna::flush_barriers(cmds);

  cmds.clearColorImage(
    filterTarget.get(),
    vk::ImageLayout::eTransferDstOptimal,
    vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
    {
      vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
      },
    });

  for (auto& image : taVarianceAndNumSamples)
  {
    cmds.clearColorImage(
      image.get(),
      vk::ImageLayout::eTransferDstOptimal,
      vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
      {
        vk::ImageSubresourceRange{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
      });
  }
}

etna::Image& SSSRPass::getReflectionTarget()
{
  return filterTarget;
}

void SSSRPass::readBlueNoise()
{
  blueNoise = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{128, 128, 1},
    .name = "BlueNoise",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .tiling = vk::ImageTiling::eOptimal,
    .layers = 64,
    .mipLevels = 1,
    .samples = vk::SampleCountFlagBits::e1,
  });

  for (size_t layer = 0; layer < 64; ++layer)
  {
    std::string path = GRAPHICS_COURSE_RESOURCES_ROOT "/textures/stbn/stbn_vec2_2Dx1D_128x128x64_" +
      std::to_string(layer) + ".png";

    int width, height, nrComponents;
    stbi_uc* data = stbi_load(path.c_str(), &width, &height, &nrComponents, 4);

    transferHelper.uploadImage(
      *oneShotCommands,
      blueNoise,
      0,
      static_cast<uint32_t>(layer),
      std::span<std::byte const>(
        reinterpret_cast<const std::byte*>(data),
        width * height * nrComponents * sizeof(std::byte)));

    stbi_image_free(data);
  }

  auto cmdBuffer = oneShotCommands->start();
  ETNA_CHECK_VK_RESULT(cmdBuffer.begin(vk::CommandBufferBeginInfo{}));

  etna::set_state(
    cmdBuffer,
    blueNoise.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmdBuffer);

  ETNA_CHECK_VK_RESULT(cmdBuffer.end());
  oneShotCommands->submitAndWait(std::move(cmdBuffer));
}
