#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <glm/glm.hpp>

#include "scene/SceneManager.hpp"
#include "render_utils/QuadRenderer.hpp"
#include "wsi/Keyboard.hpp"

#include "FramePacket.hpp"


/**
 * The meat of the sample. All things you see on the screen are contained within this class.
 * This what you want to change and expand between different samples.
 */
class WorldRenderer
{
public:
  WorldRenderer();

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
  void renderScene(vk::CommandBuffer cmd_buf, etna::ShaderProgramInfo info, bool material_pass);

private:
  enum DebugPreviewMode : uint32_t {
    DebugPreviewDisabled,
    DebugPreviewShadowMap,
    DebugPreviewDepth,
    DebugPreviewGBufferAlbedo,
    DebugPreviewGBufferMetalnessRoughness,
    DebugPreviewGBufferNorm,

    DebugPreviewModeCount
  };

  constexpr static vk::Format GBUFFER_ALBEDO_FORMAT = vk::Format::eR8G8B8A8Unorm;
  constexpr static vk::Format GBUFFER_METALNESS_ROUGHNESS_FORMAT = vk::Format::eR8G8B8A8Unorm;
  constexpr static vk::Format GBUFFER_NORM_FORMAT = vk::Format::eA2R10G10B10UnormPack32;

  // FIXME (tralf-strues): upload light data each frame using a dedicated transfer
  // instead of just mapping the buffer... could get quite large in the future, and not
  // all of us have amd gpus :)
  constexpr static uint32_t MAX_POINT_LIGHTS = 32U;

  std::unique_ptr<SceneManager> sceneMgr;

  etna::Sampler linearSampler;
  etna::Sampler pointSampler;

  glm::uvec2 resolution;

  /* Shadow Pass */
  etna::GraphicsPipeline shadowPassPipeline;

  etna::Buffer shadowCameraData;
  etna::Image shadowMap;

  /* Geometry Pass */
  etna::GraphicsPipeline geometryPassPipeline;

  etna::Buffer cameraData;
  etna::Image depth;
  etna::Image gBufferAlbedo;
  etna::Image gBufferMetalnessRoughness;
  etna::Image gBufferNorm;

  /* Deferred Pass */
  etna::ComputePipeline deferredPassPipeline;

  etna::Buffer lightData;
  etna::Image deferredTarget;

  struct PushConstantDeferredPass {
    glm::uvec2 resolution;
    glm::vec2 invResolution;
    float aspect;
    float projA;
    float projB;
  } pushConstDeferredPass;

  /* Debug Preview Pass */
  std::unique_ptr<QuadRenderer> debugPreviewRenderer;
  DebugPreviewMode debugPreviewMode = DebugPreviewDisabled;
};
