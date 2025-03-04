#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <glm/glm.hpp>

#include "Temporal.hpp"


class TAAPass
{
public:
  void loadShaders();
  void allocateResources(glm::uvec2 target_resolution, vk::Format format);
  void setupPipelines();

  etna::Image& getCurrentTarget();
  etna::Image& getResolveTarget();
  etna::Image& getMotionVectors();
  glm::vec2 getJitter();

  void resolve(vk::CommandBuffer cmd_buf, bool filter_history);

private:
  static constexpr size_t GROUP_SIZE = 8;

private:
  etna::Image& getHistory();

private:
  etna::ComputePipeline pipeline;

  etna::Sampler linearSampler;
  etna::Sampler pointSampler;

  glm::uvec2 resolution;

  Temporal<etna::Image> motionVectors;

  etna::Image currentTarget;
  Temporal<etna::Image> resolveTargets;

  size_t curJitterIdx = 0;
};
