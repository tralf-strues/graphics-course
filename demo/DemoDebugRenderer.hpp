#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <glm/glm.hpp>

#include "scene/SceneManager.hpp"
#include "wsi/Keyboard.hpp"

#include "FramePacket.hpp"

class DemoDebugRenderer
{
public:
  DemoDebugRenderer();

  void loadScene(std::filesystem::path path);

  void loadShaders();
  void allocateResources(glm::uvec2 swapchain_resolution);
  void setupPipelines(vk::Format swapchain_format);

  void debugInput(const Keyboard& kb);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(
    vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view);

private:
  etna::Image loadCubemap(std::filesystem::path path);
  etna::Buffer bakeDiffuseIrradiance(etna::Image& cubemap);
  etna::Image prefilterEnvMap(etna::Image& cubemap);

  etna::Image computeEnvBRDF();

  void renderScene(vk::CommandBuffer cmd_buf, etna::ShaderProgramInfo info, bool material_pass);

private:
  using SphericalHarmonicCoeffs = std::array<float, 27>;

  struct Environment {
    etna::Image cubemap;
    etna::Image prefilteredEnvMap;
    etna::Buffer diffuseIrradianceCoeffs;
    SphericalHarmonicCoeffs shCoeffs;
  };

  enum Method : int32_t {
    eDiffuseSH,
    eDiffuseNoPrecompute,
    eSpecularIBL
  };

  constexpr static glm::uvec2 CUBEMAP_RESOLUTION = {512, 512};
  constexpr static int32_t PREFILTERED_ENV_MAP_MIPS = 6;
  constexpr static vk::Format TEMPORAL_DIFFUSE_IRRADIANCE_FORMAT = vk::Format::eR32G32B32A32Sfloat;

  std::unique_ptr<SceneManager> sceneMgr;

  etna::Sampler linearSamplerRepeat;
  etna::Sampler linearSamplerClampToEdge;
  etna::Sampler pointSampler;

  glm::uvec2 resolution;

  etna::GraphicsPipeline demoDiffuseIndirectPipeline;
  etna::GraphicsPipeline demoDiffuseSHPipeline;
  etna::GraphicsPipeline demoSpecularIBLPipeline;
  etna::GraphicsPipeline renderCubemapPipeline;

  std::array<etna::Image, 2> temporalDiffuseIrradiance;
  size_t temporalCount = 0U;

  etna::Buffer cameraData;
  etna::Image depth;

  etna::ComputePipeline convertCubemapPipeline;
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;

  etna::ComputePipeline bakeDiffuseIrradiancePipeline;
  etna::ComputePipeline prefilterEnvMapPipeline;
  etna::ComputePipeline computeEnvBRDFPipeline;

  std::vector<Environment> environments;
  etna::Image envBRDF;

  int32_t currentEnvironmentIdx = 0;
  int32_t newEnvironmentIdx = 0;

  int32_t currentCubemapMip = 0;

  Method currentMethod = eDiffuseSH;

  float metalness = 0.4f;
  float roughness = 0.15f;
};