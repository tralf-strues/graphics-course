#include "TAAPass.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>


static constexpr std::array HALTON_SEQUENCE = {
  glm::vec2(0.510000f, 0.333333f),
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
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .name = "TAAPass::pointSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
  });

  currentTarget = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "TAAPass::currentTarget",
    .format = format,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage |
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
  });

  for (size_t i = 0; i < motionVectors.size(); ++i)
  {
    motionVectors[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "TAAPass::motionVectors[" + std::to_string(i) + "]",
      .format = vk::Format::eR16G16Sfloat,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
    });
  }

  for (size_t i = 0; i < resolveTargets.size(); ++i)
  {
    resolveTargets[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "TAAPass::resolveTargets[" + std::to_string(i) + "]",
      .format = format,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eTransferSrc,
    });
  }
}

void TAAPass::setupPipelines()
{
  pipeline = etna::get_context().getPipelineManager().createComputePipeline("taa_resolve", {});
}

etna::Image& TAAPass::getCurrentTarget()
{
  return currentTarget;
}

etna::Image& TAAPass::getResolveTarget()
{
  return resolveTargets.getCurrent();
}

etna::Image& TAAPass::getMotionVectors()
{
  return motionVectors.getCurrent();
}

glm::vec2 TAAPass::getJitter()
{
  return jitterScale * (HALTON_SEQUENCE[curJitterIdx] - 0.5f) / glm::vec2(resolution);
}

float& TAAPass::getJitterScale()
{
  return jitterScale;
}

void TAAPass::resolve(vk::CommandBuffer cmd_buf, bool filter_history)
{
  etna::set_state(
    cmd_buf,
    motionVectors.getPrevious().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf,
    motionVectors.getCurrent().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf,
    getHistory().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf,
    getCurrentTarget().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf,
    getResolveTarget().get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageWrite,
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
        motionVectors.getPrevious().genBinding(
          linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        1,
        motionVectors.getCurrent().genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        2,
        getHistory().genBinding(
          linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        3,
        getCurrentTarget().genBinding(
          pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        4,
        getResolveTarget().genBinding(
          nullptr, vk::ImageLayout::eGeneral, etna::Image::ViewParams{})),
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
    uint32_t filterHistory;
  } pushConst{
    .resolution = resolution,
    .invResolution = 1.0f / glm::vec2(resolution),
    .filterHistory = filter_history,
  };

  cmd_buf.pushConstants<PushConstant>(
    programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pushConst});

  cmd_buf.dispatch(
    (resolution.x + GROUP_SIZE - 1) / GROUP_SIZE, (resolution.y + GROUP_SIZE - 1) / GROUP_SIZE, 1);

  motionVectors.proceed();
  resolveTargets.proceed();
  curJitterIdx = (curJitterIdx + 1) % HALTON_SEQUENCE.size();
}

etna::Image& TAAPass::getHistory()
{
  return resolveTargets.getPrevious();
}