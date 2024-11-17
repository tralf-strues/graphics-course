#include "DemoDebugRenderer.hpp"

#include "shaders/CameraData.h"
#include "render_utils/Utils.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <imgui.h>
#include <stb_image.h>

constexpr std::array CUBEMAP_NAMES = {
  "Fireplace",
  "Circus Arena",
  "Small Cathedral",
};

constexpr std::array CUBEMAP_FILEPATHS = {
  GRAPHICS_COURSE_RESOURCES_ROOT "/textures/fireplace_1k.hdr",
  GRAPHICS_COURSE_RESOURCES_ROOT "/textures/circus_arena_2k.hdr",
  GRAPHICS_COURSE_RESOURCES_ROOT "/textures/small_cathedral_2k.hdr",
};

DemoDebugRenderer::DemoDebugRenderer()
  : sceneMgr{std::make_unique<SceneManager>()}
  , oneShotCommands{etna::get_context().createOneShotCmdMgr()}
  , transferHelper{etna::BlockingTransferHelper::CreateInfo{.stagingSize = 4096 * 4096 * 12}}
{
}

void DemoDebugRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  auto& ctx = etna::get_context();

  resolution = swapchain_resolution;

  linearSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "linearSampler",
    .minLod = 0.0f,
    .maxLod = vk::LodClampNone,
  });

  pointSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "pointSampler",
    .minLod = 0.0f,
    .maxLod = 0.0f,
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

  for (auto &img : temporalDiffuseIrradiance)
  {
    img = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "temporalDiffuseIrradiance",
      .format = TEMPORAL_DIFFUSE_IRRADIANCE_FORMAT,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eTransferDst,
    });
  }
}

void DemoDebugRenderer::loadScene(std::filesystem::path path)
{
  sceneMgr->selectScene(path);

  for (const auto& cubemapPath : CUBEMAP_FILEPATHS) {
    cubemaps.push_back(loadCubemap(cubemapPath));
    cubemapDiffuseIrradianceCoeffs.push_back(bakeDiffuseIrradiance(cubemaps.back()));
  }
}

void DemoDebugRenderer::loadShaders()
{
  etna::create_program(
    "demo_diffuse_indirect",
    {DEMO_SHADERS_ROOT "geometry_pass.vert.spv",
     DEMO_SHADERS_ROOT "demo_diffuse_indirect.frag.spv"});

  etna::create_program(
    "render_cubemap", {DEMO_SHADERS_ROOT "cubemap.vert.spv", DEMO_SHADERS_ROOT "cubemap.frag.spv"});

  etna::create_program("convert_cubemap", {DEMO_SHADERS_ROOT "convert_cubemap.comp.spv"});

  etna::create_program(
    "bake_diffuse_irradiance", {DEMO_SHADERS_ROOT "bake_diffuse_irradiance.comp.spv"});

  etna::create_program(
    "demo_diffuse_sh",
    {DEMO_SHADERS_ROOT "geometry_pass.vert.spv",
     DEMO_SHADERS_ROOT "demo_diffuse_sh.frag.spv"});
}

void DemoDebugRenderer::setupPipelines(vk::Format swapchain_format)
{
  etna::VertexShaderInputDescription sceneVertexInputDesc{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = sceneMgr->getVertexFormatDescription(),
    }},
  };

  auto& pipelineManager = etna::get_context().getPipelineManager();

  constexpr vk::PipelineColorBlendAttachmentState BLEND_STATE{
    .blendEnable = vk::False,
    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };

  demoDiffuseIndirectPipeline = {};
  demoDiffuseIndirectPipeline = pipelineManager.createGraphicsPipeline(
    "demo_diffuse_indirect",
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
        {.attachments = {BLEND_STATE, BLEND_STATE}, .logicOpEnable = false, .logicOp = vk::LogicOp::eAnd},
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format, TEMPORAL_DIFFUSE_IRRADIANCE_FORMAT},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });

  renderCubemapPipeline = {};
  renderCubemapPipeline = pipelineManager.createGraphicsPipeline(
    "render_cubemap",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = {},
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eNone,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .blendingConfig =
        {.attachments = {BLEND_STATE, BLEND_STATE}, .logicOpEnable = false, .logicOp = vk::LogicOp::eAnd},

      .depthConfig =
        {
          // Discard fragments that are covered by other fragments?
          .depthTestEnable = vk::True,
          .depthWriteEnable = vk::True,
          .depthCompareOp = vk::CompareOp::eLessOrEqual,
          .maxDepthBounds = 1.f,
        },

      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format, TEMPORAL_DIFFUSE_IRRADIANCE_FORMAT},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });

  convertCubemapPipeline = {};
  convertCubemapPipeline = pipelineManager.createComputePipeline("convert_cubemap", {});

  bakeDiffuseIrradiancePipeline = {};
  bakeDiffuseIrradiancePipeline =
    pipelineManager.createComputePipeline("bake_diffuse_irradiance", {});

  demoDiffuseSHPipeline = {};
  demoDiffuseSHPipeline = pipelineManager.createGraphicsPipeline(
    "demo_diffuse_sh",
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
        {.attachments = {BLEND_STATE, BLEND_STATE}, .logicOpEnable = false, .logicOp = vk::LogicOp::eAnd},
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format, TEMPORAL_DIFFUSE_IRRADIANCE_FORMAT},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });
}

etna::Image DemoDebugRenderer::loadCubemap(std::filesystem::path path)
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

  // auto mips =
  //   static_cast<uint32_t>(std::floor(std::log2(std::max(CUBEMAP_RESOLUTION.x, CUBEMAP_RESOLUTION.y)))) + 1;
  uint32_t mips = 1;

  auto cubemap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{CUBEMAP_RESOLUTION.x, CUBEMAP_RESOLUTION.y, 1},
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

  etna::set_state(
    cmdBuffer,
    environmentRect.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmdBuffer,
    cubemap.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageWrite,
    vk::ImageLayout::eGeneral,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmdBuffer);

  auto programInfo = etna::get_shader_program("convert_cubemap");

  auto descriptorSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmdBuffer,
    {
      etna::Binding(
        0,
        environmentRect.genBinding(linearSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)),

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
    .resolution = CUBEMAP_RESOLUTION,
    .invResolution = 1.0f / glm::vec2(CUBEMAP_RESOLUTION),
  };

  cmdBuffer.pushConstants<PushConstant>(
    programInfo.getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pushConst});

  cmdBuffer.dispatch((CUBEMAP_RESOLUTION.x + 15) / 16, (CUBEMAP_RESOLUTION.y + 15) / 16, 6);

  // generate_mips(cmdBuffer, cubemap);

  ETNA_CHECK_VK_RESULT(cmdBuffer.end());
  oneShotCommands->submitAndWait(std::move(cmdBuffer));

  return cubemap;
}

etna::Buffer DemoDebugRenderer::bakeDiffuseIrradiance(etna::Image& cubemap)
{
  auto& ctx = etna::get_context();

  auto coeffBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = 27 * sizeof(float),
    .bufferUsage =
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
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
    vk::PipelineBindPoint::eCompute, bakeDiffuseIrradiancePipeline.getVkPipeline());
  cmdBuffer.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    bakeDiffuseIrradiancePipeline.getVkPipelineLayout(),
    0,
    {descriptorSet.getVkSet()},
    {});

  cmdBuffer.dispatch(1, 1, 1);

  ETNA_CHECK_VK_RESULT(cmdBuffer.end());
  oneShotCommands->submitAndWait(std::move(cmdBuffer));

  return coeffBuffer;
}

void DemoDebugRenderer::debugInput(const Keyboard& kb)
{
  if (kb[KeyboardKey::kQ] == ButtonState::Falling)
  {
    // TODO: rotate cubemap
  }

  if (kb[KeyboardKey::kE] == ButtonState::Falling)
  {
    // TODO: rotate cubemap
  }

  if (kb[KeyboardKey::kR] == ButtonState::Falling)
  {
    temporalCount = 0;
  }
}

void DemoDebugRenderer::update(const FramePacket& packet)
{
  ZoneScoped;

  if (packet.cameraMoved)
  {
    temporalCount = 0;
  }

  if (newCubemapIdx != currentCubemapIdx) {
    currentCubemapIdx = newCubemapIdx;
    temporalCount = 0;
  }

  // calc main view camera
  {
    const float aspect = float(resolution.x) / float(resolution.y);

    const auto proj = packet.mainCam.projTm(aspect);

    CameraData mainCamera;
    mainCamera.view = packet.mainCam.viewTm();
    mainCamera.proj = proj;
    mainCamera.projView = proj * mainCamera.view;
    mainCamera.wsPos = packet.mainCam.position;

    mainCamera.wsForward = packet.mainCam.forward();
    mainCamera.wsRight = packet.mainCam.right();
    mainCamera.wsUp = packet.mainCam.up();

    std::memcpy(cameraData.data(), &mainCamera, sizeof(mainCamera));
  }
}

void DemoDebugRenderer::renderScene(
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

  // auto& cubemap = cubemaps[currentCubemapIdx];

  for (std::size_t instIdx = 0; instIdx < instanceMeshes.size(); ++instIdx)
  {
    // NOTE (tralf-strues): Each column of mat3 must be padded by a float, which is why mat3x4
    // is used here. Note that actually in the shader normalMatrix is declared as mat3
    // basically just for the simplicity of usage.
    struct PushConstant
    {
      glm::mat4x4 model;
      glm::mat3x4 normalMatrix;
      uint32_t temporalCount;
    } pushConst {
      .model = instanceMatrices[instIdx],
      .normalMatrix = glm::inverseTranspose(glm::mat3(instanceMatrices[instIdx])),
      .temporalCount = static_cast<uint32_t>(temporalCount)
    };

    cmd_buf.pushConstants<PushConstant>(
      info.getPipelineLayout(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, {pushConst});

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
            etna::Binding{4, cubemapDiffuseIrradianceCoeffs[currentCubemapIdx].genBinding()},
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

void DemoDebugRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  auto& cubemap = cubemaps[currentCubemapIdx];

  {
    ETNA_PROFILE_GPU(cmd_buf, demoDiffuseIndirect);

    size_t temporalInput = temporalCount % 2;
    size_t temporalOutput = (temporalCount + 1) % 2;

    if (temporalCount == 0)
    {
      etna::set_state(
        cmd_buf,
        temporalDiffuseIrradiance[temporalInput].get(),
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(cmd_buf);

      cmd_buf.clearColorImage(
        temporalDiffuseIrradiance[temporalInput].get(),
        vk::ImageLayout::eTransferDstOptimal,
        vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f},
        vk::ImageSubresourceRange{
          vk::ImageAspectFlagBits::eColor,
          0,
          1,
          0,
          1,
        });
    }

    etna::set_state(
      cmd_buf,
      temporalDiffuseIrradiance[temporalInput].get(),
      vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(cmd_buf);

    auto demoPassInfo = etna::get_shader_program("demo_diffuse_sh");
    auto cameraSet = etna::create_descriptor_set(
      demoPassInfo.getDescriptorLayoutId(0),
      cmd_buf,
      {
        etna::Binding{0, cameraData.genBinding()},
      });

    auto renderCubemapInfo = etna::get_shader_program("render_cubemap");
    auto cubemapSet = etna::create_descriptor_set(
      renderCubemapInfo.getDescriptorLayoutId(1),
      cmd_buf,
      {
        etna::Binding{
          0,
          cubemap.genBinding(
            linearSampler.get(),
            vk::ImageLayout::eShaderReadOnlyOptimal,
            etna::Image::ViewParams{
              0,
              vk::RemainingMipLevels,
              0,
              vk::RemainingArrayLayers,
              {},
              vk::ImageViewType::eCube,
            })},
      });

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},

      {
        {
          .image = target_image,
          .view = target_image_view,
          .clearColorValue = {0.0f, 0.0f, 0.0f, 0.0f},
        },

        {
          .image = temporalDiffuseIrradiance[temporalOutput].get(),
          .view = temporalDiffuseIrradiance[temporalOutput].getView({}),
          .clearColorValue = {0.0f, 0.0f, 0.0f, 0.0f},
        }
      },

      {.image = depth.get(), .view = depth.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, demoDiffuseSHPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      demoDiffuseSHPipeline.getVkPipelineLayout(),
      0,
      {cameraSet.getVkSet()},
      {});

    renderScene(cmd_buf, demoPassInfo, true);

    cameraSet = etna::create_descriptor_set(
      renderCubemapInfo.getDescriptorLayoutId(0),
      cmd_buf,
      {
        etna::Binding{0, cameraData.genBinding()},
      });

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, renderCubemapPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      renderCubemapPipeline.getVkPipelineLayout(),
      0,
      {cameraSet.getVkSet(), cubemapSet.getVkSet()},
      {});

    cmd_buf.draw(36, 1, 0, 0);
  }

  ++temporalCount;
}

void DemoDebugRenderer::drawGui()
{
  ImGui::SeparatorText("Demo Renderer");

  ImGui::Text(
    "Application average %.3f ms/frame (%.1f FPS)",
    1000.0f / ImGui::GetIO().Framerate,
    ImGui::GetIO().Framerate);

  ImGui::NewLine();

  ImGui::Combo("Cubemap", &newCubemapIdx, CUBEMAP_NAMES.data(), static_cast<int32_t>(CUBEMAP_NAMES.size()));

  ImGui::NewLine();

  ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Press 'B' to recompile and reload shaders");
  ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Press 'R' to reset diffuse irradiance accumulation");
}
