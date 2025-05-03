#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <glm/glm.hpp>

#include "cpp_glsl_compat.h"

#include "Temporal.hpp"


class SSSRPass
{
public:
  struct Params
  {
    glm::ivec2 originalResolution;
    glm::vec2 invOriginalResolution;
    glm::ivec2 resolution;
    glm::vec2 invResolution;
    uint32_t envMapMips;
    uint32_t imagePyramidMips;
    int32_t maxIterations;
    int32_t maxAccumulationSamples;
    uint32_t frameIdx;
    float depthThickness;
    float roughnessThreshold;
    float temporalStability;
    int32_t startMipLevel;
    shader_bool useBlueNoise;
    shader_bool useTemporalAccumulation;
    shader_bool useExponentialTemporalMean;
    shader_bool useFilter;
    shader_bool useTemporalVariance;
    shader_bool fallbackToAverage;
    shader_bool traceBehindSurfaces;
    shader_bool visualizeIterationCount;
  };

  void loadShaders();
  void allocateResources(glm::uvec2 target_resolution, vk::Format format);
  void setupPipelines();

  void execute(
    vk::CommandBuffer cmd_buf,
    const Params& params,
    etna::Buffer& prev_camera_buffer,
    etna::Buffer& curr_camera_buffer,
    etna::Image& hiz,
    etna::Image& prev_depth,
    Temporal<etna::Image>& gbuffer_norm,
    etna::Image& curr_motion_vectors,
    etna::Image& prev_color,
    etna::Image& gbuffer_metalness_roughness,
    const etna::Image& prefiltered_environment_map
  );

  void invalidate(vk::CommandBuffer cmd_buf);

  etna::Image& getReflectionTarget();

private:
  static constexpr int32_t GROUP_SIZE = 8;

  void readBlueNoise();

  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands{etna::get_context().createOneShotCmdMgr()};
  etna::BlockingTransferHelper transferHelper{etna::BlockingTransferHelper::CreateInfo{
    .stagingSize = 128 * 128 * 64,
  }};

  etna::Image blueNoise;

  etna::Sampler pointSampler;
  etna::Sampler linearSampler;
  etna::Sampler linearSamplerRepeat;

  etna::ComputePipeline reflectPipeline;
  etna::ComputePipeline taPipeline;
  etna::ComputePipeline filterPipeline;

  etna::Image reflectTargetReflection;
  etna::Image reflectTargetReprojectionUV;
  etna::Image avgReflection;

  etna::Image taTarget;
  Temporal<etna::Image> taVarianceAndNumSamples;

  etna::Image filterTarget;
};
