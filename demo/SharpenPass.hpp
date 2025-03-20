#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/ComputePipeline.hpp>
#include <glm/glm.hpp>


class SharpenPass
{
public:
  void loadShaders();
  void allocateResources(glm::uvec2 target_resolution, vk::Format format);
  void setupPipelines();

  etna::Image& getTarget();
  float& getAmount();

  void execute(vk::CommandBuffer cmd_buf, etna::Image& input, etna::Sampler& sampler);

private:
  static constexpr size_t GROUP_SIZE = 8;

private:
  etna::ComputePipeline pipeline;
  etna::Image target;

  glm::uvec2 resolution;
  float amount = 0.2f;
};
