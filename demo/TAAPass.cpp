#include "TAAPass.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>


static constexpr std::array HALTON_SEQUENCE = {
  glm::vec2(0.500000f, 0.333333f),
  glm::vec2(0.250000f, 0.666667f),
  glm::vec2(0.750000f, 0.111111f),
  glm::vec2(0.125000f, 0.444444f),
  glm::vec2(0.625000f, 0.777778f),
  glm::vec2(0.375000f, 0.222222f),
  glm::vec2(0.875000f, 0.555556f),
  glm::vec2(0.062500f, 0.888889f),
  glm::vec2(0.562500f, 0.037037f),
  glm::vec2(0.312500f, 0.370370f),
  glm::vec2(0.812500f, 0.703704f),
  glm::vec2(0.187500f, 0.148148f),
  glm::vec2(0.687500f, 0.481481f),
  glm::vec2(0.437500f, 0.814815f),
  glm::vec2(0.937500f, 0.259259f),
  glm::vec2(0.031250f, 0.592593f),
};

void TAAPass::loadShaders()
{
  etna::create_program("taa_resolve", {DEMO_SHADERS_ROOT "taa_resolve.comp.spv"});
}

void TAAPass::allocateResources(glm::uvec2 target_resolution, vk::Format format)
{
  resolution = target_resolution;

  auto& ctx = etna::get_context();

  linearSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .name = "TAAPass::linearSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
  });

  pointSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "TAAPass::pointSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
  });

  motionVectors = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "TAAPass::motionVectors",
    .format = vk::Format::eR16G16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  for (size_t i = 0; i < targets.size(); ++i)
  {
    targets[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "TAAPass::targets[" + std::to_string(i) + "]",
      .format = format,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
    });
  }
}

void TAAPass::setupPipelines()
{
  pipeline = etna::get_context().getPipelineManager().createComputePipeline("taa_resolve", {});
}

etna::Image& TAAPass::getTarget()
{
  return targets[curTargetIdx];
}

etna::Image& TAAPass::getMotionVectors()
{
  return motionVectors;
}

glm::vec2 TAAPass::getJitter()
{
  return (HALTON_SEQUENCE[curJitterIdx] - 0.5f) / glm::vec2(resolution);
}

void TAAPass::resolve(vk::CommandBuffer cmd_buf)
{
  etna::set_state(
    cmd_buf,
    getHistory().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf,
    getMotionVectors().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf,
    getTarget().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    vk::ImageLayout::eGeneral,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmd_buf);

  auto programInfo = etna::get_shader_program("taa_resolve");
  auto descriptorSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {
      etna::Binding(
        0,
        getHistory().genBinding(
          linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        1,
        getMotionVectors().genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        2, getTarget().genBinding(nullptr, vk::ImageLayout::eGeneral, etna::Image::ViewParams{})),
    });

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    pipeline.getVkPipelineLayout(),
    0,
    {descriptorSet.getVkSet()},
    {});

  struct PushConstant
  {
    glm::uvec2 resolution;
    glm::vec2 invResolution;
  } pushConst{
    .resolution = resolution,
    .invResolution = 1.0f / glm::vec2(resolution),
  };

  cmd_buf.pushConstants<PushConstant>(
    programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pushConst});

  cmd_buf.dispatch((resolution.x + 31) / 32, (resolution.y + 31) / 32, 1);

  curTargetIdx = (curTargetIdx + 1) % 2;
  curJitterIdx = (curJitterIdx + 1) % 16;
}

etna::Image& TAAPass::getHistory()
{
  return targets[(curTargetIdx + 1) % 2];
}