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
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .name = "SSSRPass::pointSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
  });

  linearSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .name = "SSSRPass::linearSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
  });

  reflectionTarget = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "SSSRPass::reflectionTarget",
    .format = format,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
      vk::ImageUsageFlagBits::eTransferSrc, // TODO: Remove transfer src
  });
}

void SSSRPass::setupPipelines()
{
  pipeline = etna::get_context().getPipelineManager().createComputePipeline("sssr", {});
}

void SSSRPass::execute(
  vk::CommandBuffer cmds,
  const Params& params,
  etna::Buffer& camera_buffer,
  etna::Image& hiz,
  etna::Image& gbuffer_norm,
  etna::Image& curr_motion_vectors,
  etna::Image& prev_color,
  etna::Image& gbuffer_metalness_roughness)
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
    gbuffer_norm.get(),
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
    reflectionTarget.get(),
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
      etna::Binding(0, camera_buffer.genBinding()),

      etna::Binding(
        1,
        hiz.genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        2,
        gbuffer_norm.genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        3,
        curr_motion_vectors.genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        4,
        prev_color.genBinding(
          linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        5,
        gbuffer_metalness_roughness.genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        6,
        reflectionTarget.genBinding(nullptr, vk::ImageLayout::eGeneral, etna::Image::ViewParams{})),
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
}

etna::Image& SSSRPass::getReflectionTarget()
{
  return reflectionTarget;
}
