#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/ComputePipeline.hpp>
#include <glm/glm.hpp>

class EnvironmentManager
{
public:
  struct Environment
  {
    using SHCoefficientArray = std::array<float, 27>;

    etna::Image cubemap;

    // Irradiance SH coefficients.
    etna::Buffer irradianceSHCoefficientBuffer;
    SHCoefficientArray irradianceSHCoefficientArray;

    // Prefiltered environment map for specular IBL.
    // Each mip corresponds to a roughness value.
    etna::Image prefilteredEnvMap;
  };

  struct Info
  {
    glm::uvec2 cubemapResolution{512, 512};
    int32_t prefilteredEnvMapMips{6};
  };

  constexpr static glm::uvec2 ENV_BRDF_RESOLUTION = {512, 512};

  explicit EnvironmentManager(const Info& info);

  void loadShaders() const;
  void allocateResources();
  void setupPipelines();

  void computeEnvBRDF();
  void loadEnvironment(const std::filesystem::path& path);

  glm::uvec2 getCubemapResolution() const;
  int32_t getPrefilteredEnvMapMips() const;

  std::span<const Environment> getEnvironments() const;
  etna::Image& getEnvBRDF();

private:
  etna::Image loadCubemap(const std::filesystem::path& path);
  etna::Buffer bakeDiffuseIrradiance(etna::Image& cubemap);
  etna::Image prefilterEnvMap(etna::Image& cubemap);

private:
  glm::uvec2 resolution;
  int32_t prefilteredEnvMapMips;

  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;

  etna::Sampler linearSamplerRepeat;
  etna::Sampler pointSampler;

  etna::ComputePipeline convertCubemapPipeline;
  etna::ComputePipeline bakeDiffuseIrradianceSHPipeline;
  etna::ComputePipeline prefilterEnvMapPipeline;
  etna::ComputePipeline computeEnvBRDFPipeline;

  std::vector<Environment> environments;
  etna::Image envBRDF;
};
