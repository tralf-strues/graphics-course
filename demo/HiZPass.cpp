#include "HiZPass.hpp"

#include <etna/RenderTargetStates.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>


void HiZPass::loadShaders()
{
  etna::create_program("hiz", {DEMO_SHADERS_ROOT "hiz.comp.spv"});
}

void HiZPass::allocateResources(glm::uvec2 target_resolution, uint32_t mip_levels)
{
  resolution = target_resolution;

  auto& ctx = etna::get_context();

  mipLevels = (mip_levels == FULL_MIPCHAIN)
    ? 1 + static_cast<uint32_t>(std::floor(std::log2(std::max(resolution.x, resolution.y))))
    : mip_levels;

  hiz = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "HiZ",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
    .mipLevels = mipLevels,
  });

  sampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .name = "HiZPass::sampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
  });
}

void HiZPass::setupPipelines()
{
  pipeline = etna::get_context().getPipelineManager().createComputePipeline("hiz", {});
}

etna::Image& HiZPass::getHiZ()
{
  return hiz;
}

void HiZPass::execute(vk::CommandBuffer cmds, etna::Image& depth)
{
  ETNA_PROFILE_GPU(cmds, HiZPass);

  auto programInfo = etna::get_shader_program("hiz");
  cmds.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());

  /* Transition depth to sampled */
  etna::set_state(
    cmds,
    depth.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eDepth);

  /* Transition HiZ mips [0, mipLevels - 1] to compute shader write */
  etna::set_state(
    cmds,
    hiz.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageWrite,
    vk::ImageLayout::eGeneral,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmds);

  for (uint32_t mip = 0; mip < mipLevels; ++mip)
  {
    auto& srcImage = (mip == 0) ? depth : hiz;
    auto srcMip = (mip == 0) ? 0 : mip - 1;

    glm::uvec2 srcResolution = resolution >> srcMip;
    glm::uvec2 dstResolution = resolution >> mip;

    srcResolution = glm::max(srcResolution, 1U);
    dstResolution = glm::max(dstResolution, 1U);

    auto descriptorSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(0),
      cmds,
      {
        etna::Binding(
          0,
          srcImage.genBinding(
            sampler.get(),
            vk::ImageLayout::eShaderReadOnlyOptimal,
            etna::Image::ViewParams{
              srcMip,
              1,
            })),

        etna::Binding(
          1,
          hiz.genBinding(
            nullptr,
            vk::ImageLayout::eGeneral,
            etna::Image::ViewParams{
              mip,
              1,
            })),
      },
      BarrierBehavoir::eSuppressBarriers);

    cmds.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      pipeline.getVkPipelineLayout(),
      0,
      {descriptorSet.getVkSet()},
      {});

    bool extraSrcColumn = ((srcResolution.x & 1) != 0) && (srcResolution.x != 1);
    bool extraSrcRow = ((srcResolution.y & 1) != 0) && (srcResolution.y != 1);

    struct PushConstant
    {
      glm::ivec2 srcResolution;
      glm::ivec2 dstResolution;
      glm::vec2 invSrcResolution;
      vk::Bool32 mip0;
      vk::Bool32 extraSrcColumn;
      vk::Bool32 extraSrcRow;
      vk::Bool32 extraSrcColumnAndRow;
    } pushConst{
      .srcResolution = srcResolution,
      .dstResolution = dstResolution,
      .invSrcResolution = 1.0f / glm::vec2(srcResolution),
      .mip0 = static_cast<vk::Bool32>(mip == 0),
      .extraSrcColumn = static_cast<vk::Bool32>(extraSrcColumn),
      .extraSrcRow = static_cast<vk::Bool32>(extraSrcRow),
      .extraSrcColumnAndRow = static_cast<vk::Bool32>(extraSrcColumn && extraSrcRow),
    };

    cmds.pushConstants<PushConstant>(
      programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pushConst});

    cmds.dispatch(
      (dstResolution.x + GROUP_SIZE - 1) / GROUP_SIZE,
      (dstResolution.y + GROUP_SIZE - 1) / GROUP_SIZE,
      1);

    /* Transition mip to sampled */
    vk::ImageMemoryBarrier barrier;
    barrier.setImage(hiz.get());
    barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    barrier.setOldLayout(vk::ImageLayout::eGeneral);
    barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    barrier.setSubresourceRange(vk::ImageSubresourceRange{
      vk::ImageAspectFlagBits::eColor,
      /*baseMip=*/mip,
      /*levelCount=*/1,
      0,
      1,
    });

    cmds.pipelineBarrier(
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eComputeShader,
      {},
      {},
      {},
      barrier);
  }

  /* Set state explicitly for subsequent HiZ usages with auto state tracking system to be valid */
  etna::set_state_external(
    hiz.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal);
}
