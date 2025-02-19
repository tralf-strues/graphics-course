#include "EnvironmentManager.hpp"

#include "render_utils/Utils.hpp"

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>

#include <stb_image.h>

EnvironmentManager::EnvironmentManager(const Info& info)
  : resolution(info.cubemapResolution)
  , prefilteredEnvMapMips(info.prefilteredEnvMapMips)
  , oneShotCommands{etna::get_context().createOneShotCmdMgr()}
  , transferHelper{etna::BlockingTransferHelper::CreateInfo{.stagingSize = 4096 * 4096 * 12}}
{
}

void EnvironmentManager::loadShaders() const
{
  etna::create_program("convert_cubemap", {DEMO_SHADERS_ROOT "convert_cubemap.comp.spv"});

  etna::create_program(
    "bake_diffuse_irradiance", {DEMO_SHADERS_ROOT "bake_diffuse_irradiance.comp.spv"});

  etna::create_program("prefilter_envmap", {DEMO_SHADERS_ROOT "prefilter_envmap.comp.spv"});
  etna::create_program("compute_env_brdf", {DEMO_SHADERS_ROOT "compute_env_brdf.comp.spv"});
}

void EnvironmentManager::allocateResources()
{
  linearSamplerRepeat = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "EnvironmentManager::linearSamplerRepeat",
    .minLod = 0.0f,
    .maxLod = vk::LodClampNone,
  });

  pointSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "EnvironmentManager::pointSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
  });
}

void EnvironmentManager::setupPipelines()
{
  auto& pipelineManager = etna::get_context().getPipelineManager();

  convertCubemapPipeline = pipelineManager.createComputePipeline("convert_cubemap", {});

  bakeDiffuseIrradianceSHPipeline =
    pipelineManager.createComputePipeline("bake_diffuse_irradiance", {});

  prefilterEnvMapPipeline = pipelineManager.createComputePipeline("prefilter_envmap", {});
  computeEnvBRDFPipeline = pipelineManager.createComputePipeline("compute_env_brdf", {});
}

void EnvironmentManager::computeEnvBRDF()
{
  auto& ctx = etna::get_context();

  envBRDF = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{ENV_BRDF_RESOLUTION.x, ENV_BRDF_RESOLUTION.y, 1},
    .name = "EnvironmentManager::envBRDF",
    .format = vk::Format::eR16G16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });

  auto cmdBuffer = oneShotCommands->start();
  ETNA_CHECK_VK_RESULT(cmdBuffer.begin(vk::CommandBufferBeginInfo{}));

  auto programInfo = etna::get_shader_program("compute_env_brdf");
  auto descriptorSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmdBuffer,
    {
      etna::Binding(
        0,
        envBRDF.genBinding(
          pointSampler.get(), vk::ImageLayout::eGeneral, etna::Image::ViewParams{})),
    });
  etna::flush_barriers(cmdBuffer);

  cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computeEnvBRDFPipeline.getVkPipeline());
  cmdBuffer.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    computeEnvBRDFPipeline.getVkPipelineLayout(),
    0,
    {descriptorSet.getVkSet()},
    {});

  cmdBuffer.dispatch(ENV_BRDF_RESOLUTION.x / 32, ENV_BRDF_RESOLUTION.y / 32, 1);

  etna::set_state(
    cmdBuffer,
    envBRDF.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmdBuffer);

  ETNA_CHECK_VK_RESULT(cmdBuffer.end());
  oneShotCommands->submitAndWait(std::move(cmdBuffer));
}

void EnvironmentManager::loadEnvironment(const std::filesystem::path& path)
{
  Environment environment;
  environment.cubemap = loadCubemap(path);

  environment.irradianceSHCoefficientBuffer = bakeDiffuseIrradiance(environment.cubemap);

  auto* src = environment.irradianceSHCoefficientBuffer.map();
  auto* dst = environment.irradianceSHCoefficientArray.data();
  std::memcpy(dst, src, 27 * sizeof(float));
  environment.irradianceSHCoefficientBuffer.unmap();

  environment.prefilteredEnvMap = prefilterEnvMap(environment.cubemap);

  environments.push_back(std::move(environment));
}

glm::uvec2 EnvironmentManager::getCubemapResolution() const
{
  return resolution;
}

int32_t EnvironmentManager::getPrefilteredEnvMapMips() const
{
  return prefilteredEnvMapMips;
}

std::span<const EnvironmentManager::Environment> EnvironmentManager::getEnvironments() const
{
  return environments;
}

etna::Image& EnvironmentManager::getEnvBRDF()
{
  return envBRDF;
}

etna::Image EnvironmentManager::loadCubemap(const std::filesystem::path& path)
{
  int width, height, nrComponents;
  float* data = stbi_loadf(path.string().c_str(), &width, &height, &nrComponents, 4);

  auto& ctx = etna::get_context();

  auto environmentRect = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
    .name = "environmentRect",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
      vk::ImageUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .tiling = vk::ImageTiling::eOptimal,
    .layers = 1,
    .mipLevels = 1,
    .samples = vk::SampleCountFlagBits::e1,
  });

  auto mips =
    static_cast<uint32_t>(std::floor(std::log2(std::max(resolution.x, resolution.y)))) + 1;

  auto cubemap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "cubemap",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .tiling = vk::ImageTiling::eOptimal,
    .layers = 6U,
    .mipLevels = mips,
    .samples = vk::SampleCountFlagBits::e1,
    .type = vk::ImageType::e2D,
    .flags = vk::ImageCreateFlagBits::eCubeCompatible,
  });

  transferHelper.uploadImage(
    *oneShotCommands,
    environmentRect,
    0,
    0,
    std::span<std::byte const>(
      reinterpret_cast<const std::byte*>(data), width * height * 4U * sizeof(float)));

  auto cmdBuffer = oneShotCommands->start();
  ETNA_CHECK_VK_RESULT(cmdBuffer.begin(vk::CommandBufferBeginInfo{}));

  auto programInfo = etna::get_shader_program("convert_cubemap");
  auto descriptorSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmdBuffer,
    {
      etna::Binding(
        0,
        environmentRect.genBinding(linearSamplerRepeat.get(), vk::ImageLayout::eShaderReadOnlyOptimal)),

      etna::Binding(
        1,
        cubemap.genBinding(
          nullptr,
          vk::ImageLayout::eGeneral,
          etna::Image::ViewParams{
            0,
            1,
            0,
            6,
            {},
            vk::ImageViewType::eCube,
          })),
    });
  etna::flush_barriers(cmdBuffer);

  cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, convertCubemapPipeline.getVkPipeline());
  cmdBuffer.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    convertCubemapPipeline.getVkPipelineLayout(),
    0,
    {descriptorSet.getVkSet()},
    {});

  struct PushConstant {
    glm::uvec2 resolution;
    glm::vec2 invResolution;
  } pushConst{
    .resolution = resolution,
    .invResolution = 1.0f / glm::vec2(resolution),
  };

  cmdBuffer.pushConstants<PushConstant>(
    programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pushConst});

  cmdBuffer.dispatch((resolution.x + 15) / 16, (resolution.y + 15) / 16, 6);

  generate_mips(cmdBuffer, cubemap, 6);

  etna::set_state(
    cmdBuffer,
    cubemap.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmdBuffer);

  ETNA_CHECK_VK_RESULT(cmdBuffer.end());
  oneShotCommands->submitAndWait(std::move(cmdBuffer));

  return cubemap;
}

etna::Buffer EnvironmentManager::bakeDiffuseIrradiance(etna::Image& cubemap)
{
  auto& ctx = etna::get_context();

  auto coeffBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = 27 * sizeof(float),
    .bufferUsage =
      vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "cubemapDiffuseIrradianceCoeffs",
  });

  auto cmdBuffer = oneShotCommands->start();
  ETNA_CHECK_VK_RESULT(cmdBuffer.begin(vk::CommandBufferBeginInfo{}));

  auto programInfo = etna::get_shader_program("bake_diffuse_irradiance");
  auto descriptorSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmdBuffer,
    {
      etna::Binding(
        0,
        cubemap.genBinding(
          pointSampler.get(),
          vk::ImageLayout::eShaderReadOnlyOptimal,
          etna::Image::ViewParams{
            0,
            1,
            0,
            6,
            {},
            vk::ImageViewType::eCube,
          })),

      etna::Binding(1, coeffBuffer.genBinding()),
    });
  etna::flush_barriers(cmdBuffer);

  cmdBuffer.bindPipeline(
    vk::PipelineBindPoint::eCompute, bakeDiffuseIrradianceSHPipeline.getVkPipeline());
  cmdBuffer.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    bakeDiffuseIrradianceSHPipeline.getVkPipelineLayout(),
    0,
    {descriptorSet.getVkSet()},
    {});

  cmdBuffer.dispatch(1, 1, 1);

  ETNA_CHECK_VK_RESULT(cmdBuffer.end());
  oneShotCommands->submitAndWait(std::move(cmdBuffer));

  return coeffBuffer;
}

etna::Image EnvironmentManager::prefilterEnvMap(etna::Image& cubemap)
{
  auto& ctx = etna::get_context();

  auto prefilteredEnvMap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "prefilteredEnvMap",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
      vk::ImageUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .tiling = vk::ImageTiling::eOptimal,
    .layers = 6U,
    .mipLevels = static_cast<std::size_t>(prefilteredEnvMapMips),
    .samples = vk::SampleCountFlagBits::e1,
    .type = vk::ImageType::e2D,
    .flags = vk::ImageCreateFlagBits::eCubeCompatible,
  });

  auto cmdBuffer = oneShotCommands->start();
  ETNA_CHECK_VK_RESULT(cmdBuffer.begin(vk::CommandBufferBeginInfo{}));

  auto programInfo = etna::get_shader_program("prefilter_envmap");
  cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, prefilterEnvMapPipeline.getVkPipeline());

  etna::set_state(
    cmdBuffer,
    cubemap.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferRead,
    vk::ImageLayout::eTransferSrcOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmdBuffer,
    prefilteredEnvMap.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferWrite,
    vk::ImageLayout::eTransferDstOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmdBuffer);

  /* Simply blit mip 0 */
  vk::ImageBlit blit;
  blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
  blit.srcOffsets[1] = vk::Offset3D{static_cast<int32_t>(resolution.x), static_cast<int32_t>(resolution.y), 1};
  blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
  blit.dstOffsets[1] = vk::Offset3D{static_cast<int32_t>(resolution.x), static_cast<int32_t>(resolution.y), 1};
  blit.setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 6});
  blit.setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 6});

  cmdBuffer.blitImage(
    cubemap.get(),
    vk::ImageLayout::eTransferSrcOptimal,
    prefilteredEnvMap.get(),
    vk::ImageLayout::eTransferDstOptimal,
    blit,
    vk::Filter::eLinear);

  etna::set_state(
    cmdBuffer,
    cubemap.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmdBuffer,
    prefilteredEnvMap.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageWrite,
    vk::ImageLayout::eGeneral,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmdBuffer);

  /* Calculate others */
  for (int32_t mip = 1; mip < prefilteredEnvMapMips; ++mip) {
    auto descriptorSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(0),
      cmdBuffer,
      {
        etna::Binding(
          0,
          cubemap.genBinding(
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
          1,
          prefilteredEnvMap.genBinding(
            linearSamplerRepeat.get(),
            vk::ImageLayout::eGeneral,
            etna::Image::ViewParams{
              static_cast<uint32_t>(mip),
              1,
              0,
              vk::RemainingArrayLayers,
              {},
              vk::ImageViewType::eCube,
            })),
      });

    cmdBuffer.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      prefilterEnvMapPipeline.getVkPipelineLayout(),
      0,
      {descriptorSet.getVkSet()},
      {});

    glm::uvec2 res = {resolution.x >> mip, resolution.y >> mip};

    struct PushConstant
    {
      glm::uvec2 resolution;
      glm::vec2 invResolution;
      int32_t mip;
      float roughness;
    } pushConst{
      .resolution = res,
      .invResolution = 1.0f / glm::vec2(res),
      .mip = mip,
      .roughness = static_cast<float>(mip) / (prefilteredEnvMapMips - 1.0f),
    };

    cmdBuffer.pushConstants<PushConstant>(
      programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pushConst});

    cmdBuffer.dispatch((res.x + 15) / 16, (res.y + 15) / 16, 6);
  }

  etna::set_state(
    cmdBuffer,
    prefilteredEnvMap.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmdBuffer);

  ETNA_CHECK_VK_RESULT(cmdBuffer.end());
  oneShotCommands->submitAndWait(std::move(cmdBuffer));

  return prefilteredEnvMap;
}
