#ifndef SSSR_COMMON_GLSL
#define SSSR_COMMON_GLSL

#include "CameraData.h"
#include "Random.glsl"
#include "PBR.glsl"

layout(push_constant) uniform params_t
{
  ivec2 resolution;
  vec2 invResolution;

  uint envMapMips;
  int maxIterations;

  uint frameIdx;

  float depthThickness;
  float roughnessThreshold;

  int startMip;
  bool useTemporalAccumulation;
  bool useFilter;
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
