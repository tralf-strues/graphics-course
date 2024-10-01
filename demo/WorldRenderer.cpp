#include "WorldRenderer.hpp"

#include "shaders/CameraData.h"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <imgui.h>


WorldRenderer::WorldRenderer()
  : sceneMgr{std::make_unique<SceneManager>()}
{
}

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  auto& ctx = etna::get_context();

  resolution = swapchain_resolution;

  linearSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "linearSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f, // FIXME (tralf-strues): use mip mapping
  });

  pointSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "pointSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
  });

  /* Shadow Pass */
  shadowCameraData = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(CameraData),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "shadowCameraData"});
  shadowCameraData.map();

  shadowMap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{2048, 2048, 1},
    .name = "shadow_map",
    .format = vk::Format::eD16Unorm,
    .imageUsage =
      vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  /* Geometry Pass */
  cameraData = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(CameraData),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "cameraData"});
  cameraData.map();

  depth = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage =
      vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  gBufferAlbedo = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "gBufferAlbedo",
    .format = GBUFFER_ALBEDO_FORMAT,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  gBufferMetalnessRoughness = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "gBufferMetalnessRoughness",
    .format = GBUFFER_METALNESS_ROUGHNESS_FORMAT,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  gBufferNorm = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "gBufferNorm",
    .format = GBUFFER_NORM_FORMAT,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  /* Deferred Pass */
  deferredTarget = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "deferredTarget",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage,
  });

  lightData = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(DirectionalLight) + sizeof(uint32_t) + MAX_POINT_LIGHTS * sizeof(PointLight),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "lightData"});
  lightData.map();
}

void WorldRenderer::loadScene(std::filesystem::path path)
{
  sceneMgr->selectScene(path);
}

void WorldRenderer::loadShaders()
{
  etna::create_program("shadow_pass", {DEMO_SHADERS_ROOT "geometry_pass.vert.spv"});

  etna::create_program(
    "geometry_pass",
    {DEMO_SHADERS_ROOT "geometry_pass.vert.spv", DEMO_SHADERS_ROOT "geometry_pass.frag.spv"});

  etna::create_program("deferred_pass", {DEMO_SHADERS_ROOT "deferred_pass.comp.spv"});
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  debugPreviewRenderer = std::make_unique<QuadRenderer>(QuadRenderer::CreateInfo{
    .format = swapchain_format,
    .rect = {{0, 0}, {512, 512}},
  });

  etna::VertexShaderInputDescription sceneVertexInputDesc{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = sceneMgr->getVertexFormatDescription(),
    }},
  };

  auto& pipelineManager = etna::get_context().getPipelineManager();

  shadowPassPipeline = {};
  shadowPassPipeline = pipelineManager.createGraphicsPipeline(
    "shadow_pass",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = sceneVertexInputDesc,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eBack,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = vk::Format::eD16Unorm,
        },
    });

  constexpr vk::PipelineColorBlendAttachmentState BLEND_STATE{
    .blendEnable = vk::False,
    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };

  geometryPassPipeline = {};
  geometryPassPipeline = pipelineManager.createGraphicsPipeline(
    "geometry_pass",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = sceneVertexInputDesc,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eBack,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .blendingConfig =
        {.attachments = {BLEND_STATE, BLEND_STATE, BLEND_STATE},
         .logicOpEnable = false,
         .logicOp = vk::LogicOp::eAnd},
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats =
            {GBUFFER_ALBEDO_FORMAT, GBUFFER_METALNESS_ROUGHNESS_FORMAT, GBUFFER_NORM_FORMAT},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });

  deferredPassPipeline = {};
  deferredPassPipeline = pipelineManager.createComputePipeline("deferred_pass", {});
}

void WorldRenderer::debugInput(const Keyboard& kb)
{
  if (kb[KeyboardKey::kQ] == ButtonState::Falling)
  {
    debugPreviewMode =
      static_cast<DebugPreviewMode>((debugPreviewMode + 1U) % DebugPreviewModeCount);
  }

  // if (kb[KeyboardKey::kP] == ButtonState::Falling)
  //   lightProps.usePerspectiveM = !lightProps.usePerspectiveM;
}

void WorldRenderer::update(const FramePacket& packet)
{
  ZoneScoped;

  // update light data
  {
    auto* dst = lightData.data();

    std::memcpy(dst, &packet.dirLight, sizeof(packet.dirLight));
    dst += sizeof(packet.dirLight);

    const auto pointLightCount = static_cast<uint32_t>(packet.pointLights.size());
    std::memcpy(dst, &pointLightCount, sizeof(pointLightCount));
    dst += sizeof(pointLightCount);

    dst += 3U * sizeof(uint32_t);  // Padding to the vec4 alignment before the array of structs

    std::memcpy(dst, packet.pointLights.data(), packet.pointLights.size_bytes());
  }

  // calc shadow camera
  {
    const auto proj = glm::orthoLH_ZO(+10.0f, -10.0f, +10.0f, -10.0f, 0.0f, 24.0f);

    Camera shadowCam;
    shadowCam.lookAt({-8, 10, 8}, {0, 0, 0}, {0, 1, 0});

    CameraData shadowCamera;
    shadowCamera.view = shadowCam.viewTm();
    shadowCamera.projView = proj * shadowCamera.view;
    shadowCamera.wsPos = glm::vec4(shadowCam.position, 0.0f);

    std::memcpy(shadowCameraData.data(), &shadowCamera, sizeof(shadowCamera));
  }

  // calc main view camera
  {
    const float aspect = float(resolution.x) / float(resolution.y);

    const auto proj = packet.mainCam.projTm(aspect);

    CameraData mainCamera;
    mainCamera.view = packet.mainCam.viewTm();
    mainCamera.projView = proj * mainCamera.view;
    mainCamera.wsPos = packet.mainCam.position;

    pushConstDeferredPass.proj22 = proj[2][2];
    pushConstDeferredPass.proj23 = proj[3][2];
    pushConstDeferredPass.invProj00 = 1.0f / proj[0][0];
    pushConstDeferredPass.invProj11 = 1.0f / proj[1][1];

    mainCamera.wsForward = packet.mainCam.forward();
    mainCamera.wsRight = packet.mainCam.right();
    mainCamera.wsUp = packet.mainCam.up();

    std::memcpy(cameraData.data(), &mainCamera, sizeof(mainCamera));
  }
}

void WorldRenderer::renderScene(
  vk::CommandBuffer cmd_buf, etna::ShaderProgramInfo info, bool material_pass)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  cmd_buf.bindVertexBuffers(0, {sceneMgr->getVertexBuffer()}, {0});
  cmd_buf.bindIndexBuffer(sceneMgr->getIndexBuffer(), 0, vk::IndexType::eUint32);

  auto instanceMeshes = sceneMgr->getInstanceMeshes();
  auto instanceMatrices = sceneMgr->getInstanceMatrices();

  auto meshes = sceneMgr->getMeshes();
  auto relems = sceneMgr->getRenderElements();

  for (std::size_t instIdx = 0; instIdx < instanceMeshes.size(); ++instIdx)
  {
    struct PushConstant
    {
      glm::mat4x4 model;
      glm::mat3x4 normalMatrix;
    } pushConst{
      .model = instanceMatrices[instIdx],
      .normalMatrix = glm::inverseTranspose(instanceMatrices[instIdx])};

    cmd_buf.pushConstants<PushConstant>(
      info.getPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, {pushConst});

    const auto meshIdx = instanceMeshes[instIdx];

    for (std::size_t j = 0; j < meshes[meshIdx].relemCount; ++j)
    {
      const auto relemIdx = meshes[meshIdx].firstRelem + j;
      const auto& relem = relems[relemIdx];
      const auto& mat = *relem.material;

      if (material_pass)
      {
        auto materialSet = etna::create_descriptor_set(
          info.getDescriptorLayoutId(1),
          cmd_buf,
          {
            etna::Binding{
              0,
              mat.texAlbedo->genBinding(
                linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
            etna::Binding{
              1,
              mat.texMetalnessRoughness->genBinding(
                linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
            etna::Binding{
              2,
              mat.texNorm->genBinding(
                linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
            etna::Binding{
              3,
              mat.texEmissive->genBinding(
                linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
          });

        cmd_buf.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          info.getPipelineLayout(),
          1,
          {materialSet.getVkSet()},
          {});
      }

      cmd_buf.drawIndexed(relem.indexCount, 1, relem.indexOffset, relem.vertexOffset, 0);
    }
  }
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf,
  [[maybe_unused]] vk::Image target_image,
  [[maybe_unused]] vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  // Shadow Pass
  {
    ETNA_PROFILE_GPU(cmd_buf, shadowPass);

    auto shadowPassInfo = etna::get_shader_program("shadow_pass");
    auto cameraSet = etna::create_descriptor_set(
      shadowPassInfo.getDescriptorLayoutId(0),
      cmd_buf,
      {
        etna::Binding{0, shadowCameraData.genBinding()},
      });

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {2048, 2048}},
      {},
      {.image = shadowMap.get(), .view = shadowMap.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, shadowPassPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      shadowPassPipeline.getVkPipelineLayout(),
      0,
      {cameraSet.getVkSet()},
      {});

    renderScene(cmd_buf, shadowPassInfo, false);
  }

  // Geometry Pass
  {
    ETNA_PROFILE_GPU(cmd_buf, geometryPass);

    auto geometryPassInfo = etna::get_shader_program("geometry_pass");
    auto cameraSet = etna::create_descriptor_set(
      geometryPassInfo.getDescriptorLayoutId(0),
      cmd_buf,
      {
        etna::Binding{0, cameraData.genBinding()},
      });

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},

      {
        {
          .image = gBufferAlbedo.get(),
          .view = gBufferAlbedo.getView({}),
          .clearColorValue = {0.0f, 0.0f, 0.0f, 0.0f},
        },
        {
          .image = gBufferMetalnessRoughness.get(),
          .view = gBufferMetalnessRoughness.getView({}),
          .clearColorValue = {0.0f, 0.0f, 0.0f, 0.0f},
        },
        {
          .image = gBufferNorm.get(),
          .view = gBufferNorm.getView({}),
          .clearColorValue = {0.0f, 0.0f, 0.0f, 0.0f},
        },
      },

      {.image = depth.get(), .view = depth.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, geometryPassPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      geometryPassPipeline.getVkPipelineLayout(),
      0,
      {cameraSet.getVkSet()},
      {});

    renderScene(cmd_buf, geometryPassInfo, true);
  }

  // Deferred Pass
  {
    ETNA_PROFILE_GPU(cmd_buf, deferredPass);

    etna::set_state(
      cmd_buf,
      deferredTarget.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmd_buf,
      gBufferAlbedo.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmd_buf,
      gBufferMetalnessRoughness.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmd_buf,
      gBufferNorm.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmd_buf,
      depth.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eDepth);

    etna::set_state(
      cmd_buf,
      shadowMap.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eDepth);

    etna::flush_barriers(cmd_buf);

    auto deferredPassInfo = etna::get_shader_program("deferred_pass");

    auto cameraSet = etna::create_descriptor_set(
      deferredPassInfo.getDescriptorLayoutId(0),
      cmd_buf,
      {
        etna::Binding{0, cameraData.genBinding()},
      });

    auto resourceSet = etna::create_descriptor_set(
      deferredPassInfo.getDescriptorLayoutId(1),
      cmd_buf,
      {
        etna::Binding(0, deferredTarget.genBinding(nullptr, vk::ImageLayout::eGeneral)),
        etna::Binding(
          1, gBufferAlbedo.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)),
        etna::Binding(
          2,
          gBufferMetalnessRoughness.genBinding(
            pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)),
        etna::Binding(
          3, gBufferNorm.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)),
        etna::Binding(
          4, depth.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)),
        etna::Binding(
          5, shadowMap.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)),
        etna::Binding(6, shadowCameraData.genBinding()),
        etna::Binding(7, lightData.genBinding()),
      });

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, deferredPassPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      deferredPassPipeline.getVkPipelineLayout(),
      0,
      {cameraSet.getVkSet(), resourceSet.getVkSet()},
      {});

    pushConstDeferredPass.resolution = resolution;
    pushConstDeferredPass.invResolution = 1.0f / glm::vec2(resolution);

    cmd_buf.pushConstants<PushConstantDeferredPass>(
      deferredPassInfo.getPipelineLayout(),
      vk::ShaderStageFlagBits::eCompute,
      0,
      {pushConstDeferredPass});

    cmd_buf.dispatch((resolution.x + 15) / 16, (resolution.y + 15) / 16, 1);
  }

  // Blit from target to swapchain image
  {
    etna::set_state(
      cmd_buf,
      deferredTarget.get(),
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferRead,
      vk::ImageLayout::eTransferSrcOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::set_state(
      cmd_buf,
      target_image,
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eTransferDstOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::flush_barriers(cmd_buf);

    vk::ImageBlit blitInfo{
      .srcSubresource =
        {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = 0,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
      .srcOffsets =
        {
          {{{0, 0, 0},
            {static_cast<int32_t>(resolution.x), static_cast<int32_t>(resolution.y), 1}}}},
      .dstSubresource =
        {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = 0,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
      .dstOffsets =
        {
          {{{0, 0, 0},
            {static_cast<int32_t>(resolution.x), static_cast<int32_t>(resolution.y), 1}}}},
    };

    cmd_buf.blitImage(
      deferredTarget.get(),
      vk::ImageLayout::eTransferSrcOptimal,
      target_image,
      vk::ImageLayout::eTransferDstOptimal,
      blitInfo,
      vk::Filter::eNearest);

    etna::set_state(
      cmd_buf,
      target_image,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      vk::AccessFlagBits2::eColorAttachmentWrite,
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageAspectFlagBits::eColor);

    etna::flush_barriers(cmd_buf);
  }

  switch (debugPreviewMode)
  {
  case DebugPreviewDisabled: {
    break;
  }

  case DebugPreviewShadowMap: {
    debugPreviewRenderer->render(
      cmd_buf, target_image, target_image_view, shadowMap, linearSampler);
    break;
  }

  case DebugPreviewDepth: {
    debugPreviewRenderer->render(cmd_buf, target_image, target_image_view, depth, linearSampler);
    break;
  }

  case DebugPreviewGBufferAlbedo: {
    debugPreviewRenderer->render(
      cmd_buf, target_image, target_image_view, gBufferAlbedo, linearSampler);
    break;
  }

  case DebugPreviewGBufferMetalnessRoughness: {
    debugPreviewRenderer->render(
      cmd_buf, target_image, target_image_view, gBufferMetalnessRoughness, linearSampler);
    break;
  }

  case DebugPreviewGBufferNorm: {
    debugPreviewRenderer->render(
      cmd_buf, target_image, target_image_view, gBufferNorm, linearSampler);
    break;
  }

  default: {
    break;
  }
  }
}

void WorldRenderer::drawGui()
{
  ImGui::Begin("Demo settings");

  // float color[3]{uniformParams.baseColor.r, uniformParams.baseColor.g,
  // uniformParams.baseColor.b}; ImGui::ColorEdit3(
  //   "Meshes base color", color, ImGuiColorEditFlags_PickerHueWheel |
  //   ImGuiColorEditFlags_NoInputs);
  // uniformParams.baseColor = {color[0], color[1], color[2]};

  // float pos[3]{uniformParams.lightPos.x, uniformParams.lightPos.y, uniformParams.lightPos.z};
  // ImGui::SliderFloat3("Light source position", pos, -10.f, 10.f);
  // uniformParams.lightPos = {pos[0], pos[1], pos[2]};

  ImGui::Text(
    "Application average %.3f ms/frame (%.1f FPS)",
    1000.0f / ImGui::GetIO().Framerate,
    ImGui::GetIO().Framerate);

  ImGui::NewLine();

  ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Press 'B' to recompile and reload shaders");
  ImGui::End();
}
