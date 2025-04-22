#include "SSSRPass.hpp"

#include <etna/RenderTargetStates.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>


void SSSRPass::loadShaders()
{
  etna::create_program("sssr", {DEMO_SHADERS_ROOT "sssr.comp.spv"});
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

  for (size_t i = 0; i < reflectionTarget.size(); ++i)
  {
    reflectionTarget[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "SSSRPass::reflectionTarget[" + std::to_string(i) + "]",
      .format = format,
      .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst, // TODO: Remove transfer src
    });
  }
}

void SSSRPass::setupPipelines()
{
  pipeline = etna::get_context().getPipelineManager().createComputePipeline("sssr", {});
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
  ETNA_PROFILE_GPU(cmds, SSSRPass);

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

  etna::set_state(
    cmds,
    reflectionTarget.getPrevious().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmds,
    reflectionTarget.getCurrent().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageWrite,
    vk::ImageLayout::eGeneral,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmds);

  cmds.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());

  auto programInfo = etna::get_shader_program("sssr");
  auto descriptorSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmds,
    {
      etna::Binding(0, prev_camera_buffer.genBinding()),
      etna::Binding(1, curr_camera_buffer.genBinding()),

      etna::Binding(
        2,
        hiz.genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        3,
        prev_depth.genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        4,
        gbuffer_norm.getPrevious().genBinding(
          linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        5,
        gbuffer_norm.getCurrent().genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        6,
        curr_motion_vectors.genBinding(
          linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        7,
        prev_color.genBinding(
          linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        8,
        gbuffer_metalness_roughness.genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        9,
        prefiltered_environment_map.genBinding(
          linearSamplerRepeat.get(),
          vk::ImageLayout::eShaderReadOnlyOptimal,
          etna::Image::ViewParams{
            0,
            vk::RemainingMipLevels,
            0,
            vk::RemainingArrayLayers,
            {},
            vk::ImageViewType::eCube,
          })),

      etna::Binding(
        10,
        reflectionTarget.getPrevious().genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        11,
        reflectionTarget.getCurrent().genBinding(
          nullptr, vk::ImageLayout::eGeneral, etna::Image::ViewParams{})),
    });

  cmds.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());
  cmds.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    pipeline.getVkPipelineLayout(),
    0,
    {descriptorSet.getVkSet()},
    {});

  cmds.pushConstants<Params>(
    programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {params});

  cmds.dispatch(
    (resolution.x + GROUP_SIZE - 1) / GROUP_SIZE, (resolution.y + GROUP_SIZE - 1) / GROUP_SIZE, 1);

  reflectionTarget.proceed();
}

void SSSRPass::invalidate(vk::CommandBuffer cmds)
{
  for (auto& image : reflectionTarget)
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

  for (auto& image : reflectionTarget)
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
  return reflectionTarget.getCurrent();
}
