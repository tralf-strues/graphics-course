#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/ComputePipeline.hpp>
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
    int32_t maxIterations;
    uint32_t frameIdx;
    float depthThickness;
    float roughnessThreshold;
    int32_t startMipLevel;
    shader_bool useTemporalAccumulation;
    shader_bool useFilter;
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

private:
  etna::Sampler pointSampler;
  etna::Sampler linearSampler;
  etna::Sampler linearSamplerRepeat;

  etna::ComputePipeline reflectPipeline;
  etna::ComputePipeline taPipeline;
  etna::ComputePipeline filterPipeline;

  // Temporal<etna::Image> reflectionTarget;

  etna::Image reflectTargetReflection;
  etna::Image reflectTargetReprojectionUV;

  etna::Image taTarget;
  etna::Image filterTarget;
};
