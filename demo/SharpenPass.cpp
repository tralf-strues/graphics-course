#include "SharpenPass.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>


void SharpenPass::loadShaders()
{
  etna::create_program("sharpen", {DEMO_SHADERS_ROOT "sharpen.comp.spv"});
}

void SharpenPass::allocateResources(glm::uvec2 target_resolution, vk::Format format)
{
  resolution = target_resolution;

  auto& ctx = etna::get_context();

  target = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "SharpenPass::target",
    .format = format,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
  });
}

void SharpenPass::setupPipelines()
{
  pipeline = etna::get_context().getPipelineManager().createComputePipeline("sharpen", {});
}

etna::Image& SharpenPass::getTarget()
{
  return target;
}

float& SharpenPass::getAmount()
{
  return amount;
}

void SharpenPass::execute(vk::CommandBuffer cmd_buf, etna::Image& input, etna::Sampler& sampler)
{
  etna::set_state(
    cmd_buf,
    input.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf,
    target.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageWrite,
    vk::ImageLayout::eGeneral,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmd_buf);

  auto programInfo = etna::get_shader_program("sharpen");
  auto descriptorSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {
      etna::Binding(
        0,
        input.genBinding(
          sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, etna::Image::ViewParams{})),

      etna::Binding(
        1, target.genBinding(nullptr, vk::ImageLayout::eGeneral, etna::Image::ViewParams{})),
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
    float amount;
  } pushConst{
    .resolution = resolution,
    .amount = amount,
  };

  cmd_buf.pushConstants<PushConstant>(
    programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pushConst});

  cmd_buf.dispatch(
    (resolution.x + GROUP_SIZE - 1) / GROUP_SIZE, (resolution.y + GROUP_SIZE - 1) / GROUP_SIZE, 1);
}
