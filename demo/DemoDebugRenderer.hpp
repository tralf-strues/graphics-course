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
  void loadCubemap(std::filesystem::path path);

  void renderScene(vk::CommandBuffer cmd_buf, etna::ShaderProgramInfo info, bool material_pass);

private:
  constexpr static glm::uvec2 CUBEMAP_RESOLUTION = {1024U, 1024U};

  std::unique_ptr<SceneManager> sceneMgr;

  etna::Sampler linearSampler;
  etna::Sampler pointSampler;

  glm::uvec2 resolution;

  etna::GraphicsPipeline demoDiffuseIndirectPipeline;
  etna::GraphicsPipeline renderCubemapPipeline;

  etna::Buffer cameraData;
  etna::Image depth;

  etna::ComputePipeline convertCubemapPipeline;
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;
  etna::Image cubemap;
};