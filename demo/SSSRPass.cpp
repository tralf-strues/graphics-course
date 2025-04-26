#include "SSSRPass.hpp"

#include <etna/RenderTargetStates.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>

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

void SSSRPass::allocateResources(glm::uvec2 target_resolution, vk::Format format)
{
  resolution = target_resolution;

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
    .maxLod = 0.0f,
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
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
  });

  reflectTargetReprojectionUV = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "SSSRPass::reflectTargetReprojectionUV",
    .format = vk::Format::eR16G16Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });

  taTarget = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "SSSRPass::taTarget",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });

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
        binding_sampled(4, gbuffer_norm.getCurrent(), pointSampler),
        binding_sampled(5, curr_motion_vectors, linearSampler),
        binding_sampled(6, prev_color, linearSampler),
        binding_sampled(7, gbuffer_metalness_roughness, pointSampler),
        binding_sampled_cube(8, prefiltered_environment_map, linearSamplerRepeat),
        binding_write(9, reflectTargetReflection),
        binding_write(10, reflectTargetReprojectionUV),
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
      (resolution.x + GROUP_SIZE - 1) / GROUP_SIZE,
      (resolution.y + GROUP_SIZE - 1) / GROUP_SIZE,
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
      filterTarget.get(),
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

    etna::flush_barriers(cmds);

    cmds.bindPipeline(vk::PipelineBindPoint::eCompute, taPipeline.getVkPipeline());

    auto programInfo = etna::get_shader_program("sssr_ta");
    auto descriptorSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(0),
      cmds,
      {
        binding_sampled(0, hiz, pointSampler),
        binding_sampled(1, prev_depth, pointSampler),
        binding_sampled(2, gbuffer_norm.getPrevious(), linearSampler),
        binding_sampled(3, gbuffer_norm.getCurrent(), pointSampler),
        binding_sampled(4, gbuffer_metalness_roughness, pointSampler),
        binding_sampled(5, filterTarget, linearSampler),
        binding_sampled(6, reflectTargetReflection, pointSampler),
        binding_sampled(7, reflectTargetReprojectionUV, pointSampler),
        binding_write(8, taTarget),
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
      (resolution.x + GROUP_SIZE - 1) / GROUP_SIZE,
      (resolution.y + GROUP_SIZE - 1) / GROUP_SIZE,
      1);
  }

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
      filterTarget.get(),
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
        binding_write(5, filterTarget),
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
      (resolution.x + GROUP_SIZE - 1) / GROUP_SIZE,
      (resolution.y + GROUP_SIZE - 1) / GROUP_SIZE,
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
}

etna::Image& SSSRPass::getReflectionTarget()
{
  return filterTarget;
}
