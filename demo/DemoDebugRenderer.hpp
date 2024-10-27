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

  void renderScene(vk::CommandBuffer cmd_buf, etna::ShaderProgramInfo info, bool material_pass);

private:
  constexpr static glm::uvec2 CUBEMAP_RESOLUTION = {512U, 512U};
  constexpr static vk::Format TEMPORAL_DIFFUSE_IRRADIANCE_FORMAT = vk::Format::eR32G32B32A32Sfloat;

  std::unique_ptr<SceneManager> sceneMgr;

  etna::Sampler linearSampler;
  etna::Sampler pointSampler;

  glm::uvec2 resolution;

  etna::GraphicsPipeline demoDiffuseIndirectPipeline;
  etna::GraphicsPipeline renderCubemapPipeline;

  std::array<etna::Image, 2> temporalDiffuseIrradiance;
  size_t temporalCount = 0U;

  etna::Buffer cameraData;
  etna::Image depth;

  etna::ComputePipeline convertCubemapPipeline;
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;

  std::vector<etna::Image> cubemaps;
  int32_t currentCubemapIdx = 0;
  int32_t newCubemapIdx = 0;
};