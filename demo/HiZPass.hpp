#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/ComputePipeline.hpp>
#include <glm/glm.hpp>


class HiZPass
{
public:
  constexpr static uint32_t FULL_MIPCHAIN = std::numeric_limits<uint32_t>::max();

  void loadShaders();
  void allocateResources(glm::uvec2 target_resolution, uint32_t mip_levels);
  void setupPipelines();

  etna::Image& getHiZ();

  void execute(vk::CommandBuffer cmd_buf, etna::Image& depth);

private:
  static constexpr size_t GROUP_SIZE = 8;

private:
  etna::ComputePipeline pipeline;
  etna::Image hiz;
  etna::Sampler sampler;

  glm::uvec2 resolution;
  uint32_t mipLevels = FULL_MIPCHAIN;
};
