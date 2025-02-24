#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <glm/glm.hpp>


class TAAPass
{
public:
  void loadShaders();
  void allocateResources(glm::uvec2 resolution, vk::Format format);
  void setupPipelines();

  etna::Image& getTarget();
  etna::Image& getMotionVectors();
  glm::vec2 getJitter();

  void resolve(vk::CommandBuffer cmd_buf);

private:
  etna::Image& getHistory();

private:
  etna::ComputePipeline pipeline;

  etna::Sampler linearSampler;
  etna::Sampler pointSampler;

  glm::uvec2 resolution;

  etna::Image motionVectors;

  std::array<etna::Image, 2> targets;

  size_t curTargetIdx = 0;
  size_t curJitterIdx = 0;
};
