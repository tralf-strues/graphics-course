#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/ComputePipeline.hpp>
#include <glm/glm.hpp>

#include "cpp_glsl_compat.h"

class SSSRPass
{
public:
  struct Params
  {
    glm::uvec2 resolution;
    glm::vec2 invResolution;
    float proj22;
    float proj23;
    float invProj00;
    float invProj11;
    int32_t maxIterations;
    int32_t samplesPerFrame;
    float depthThickness;
    int32_t startMipLevel;
    shader_bool traceBehindSurfaces;
    shader_bool useWorldSpaceHitConfidence;
    shader_bool visualizeIterationCount;
  };

  void loadShaders();
  void allocateResources(glm::uvec2 target_resolution, vk::Format format);
  void setupPipelines();

  void execute(
    vk::CommandBuffer cmd_buf,
    const Params& params,
    etna::Buffer& camera_buffer,
    etna::Image& hiz,
    etna::Image& gbuffer_norm,
    etna::Image& curr_motion_vectors,
    etna::Image& prev_color,
    etna::Image& gbuffer_metalness_roughness);

  etna::Image& getReflectionTarget();

private:
  static constexpr size_t GROUP_SIZE = 8;

private:
  etna::ComputePipeline pipeline;
  etna::Sampler pointSampler;
  etna::Sampler linearSampler;

  etna::Image reflectionTarget;

  glm::uvec2 resolution;
};
