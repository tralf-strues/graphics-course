#ifndef SSSR_COMMON_GLSL
#define SSSR_COMMON_GLSL

#include "CameraData.h"
#include "Random.glsl"
#include "PBR.glsl"

layout(push_constant) uniform params_t
{
  ivec2 originalResolution;
  vec2 invOriginalResolution;

  ivec2 resolution;
  vec2 invResolution;

  uint envMapMips;
  uint imagePyramidMips;
  int maxIterations;
  int maxAccumulationSamples;

  uint frameIdx;

  float depthThickness;
  float roughnessThreshold;
  float temporalStability;

  int startMip;
  bool useBlueNoise;
  bool useTemporalAccumulation;
  bool useExponentialTemporalMean;
  bool useFilter;
  bool useTemporalVariance;
  bool fallbackToAverage;
  bool traceBehindSurfaces;
  bool visualizeIterationCount;
} params;

bool CheckRoughnessThreshold(float roughness)
{
  return roughness < params.roughnessThreshold;
}

bool MirrorReflection(float roughness)
{
  return roughness <= 0.001f;
}

#endif // SSSR_COMMON_GLSL
