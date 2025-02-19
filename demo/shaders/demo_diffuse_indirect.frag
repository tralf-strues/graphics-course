#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "CameraData.h"
#include "NormalPerturbation.glsl"
#include "PBR.glsl"
#include "Random.glsl"

//==================================================================================================
// Descriptor bindings / push constants
//--------------------------------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform camera_data_t
{
  CameraData camera;
};

layout(set = 1, binding = 0) uniform sampler2D texAlbedo;
layout(set = 1, binding = 1) uniform sampler2D texMetalnessRoughness;
layout(set = 1, binding = 2) uniform sampler2D texNorm;
layout(set = 1, binding = 3) uniform sampler2D texEmissive;

layout(set = 1, binding = 4) uniform samplerCube cubemap;
layout(set = 1, binding = 5) uniform sampler2D temporalDiffuseIrradiance;

layout(push_constant) uniform params_t
{
  mat4 model;
  mat3 normalMatrix;
  uint temporalCount;
  float metalness;
  float roughness;
} params;
//==================================================================================================

//==================================================================================================
// Stage linkage
//--------------------------------------------------------------------------------------------------
layout(location = 0) in vs_out_t
{
  vec3 wsPos;
  vec3 wsNorm;
  vec2 texCoord;
} vertex;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_temporalDiffuseIrradiance;
//==================================================================================================

void main()
{
  SurfacePoint point;
  point.normal    = PerturbNormal(texNorm, normalize(vertex.wsNorm), vertex.wsPos, vertex.texCoord);
  point.position  = vertex.wsPos;
  // point.toCam     = normalize(camera.wsPos - point.position);
  point.albedo    = texture(texAlbedo, vertex.texCoord).rgb;
  // point.metalness = texture(texMetalnessRoughness, vertex.texCoord).b;
  // point.roughness = texture(texMetalnessRoughness, vertex.texCoord).g;
  // point.f0        = vec3(0.04f);
  // point.f0        = mix(point.f0, point.albedo, point.metalness);

  uint seed = (uint(gl_FragCoord.x) * uint(1973) + uint(gl_FragCoord.y) * uint(9277) +
    uint(params.temporalCount) * uint(26699)) | uint(1);

  vec2 resolution     = vec2(textureSize(temporalDiffuseIrradiance, 0).xy);
  vec2 fragUV         = gl_FragCoord.xy / resolution;
  vec3 prevIrradiance = texture(temporalDiffuseIrradiance, fragUV).rgb;

  const uint SAMPLES = 16;

  vec3 newIrradiance = vec3(0.0f);
  for (uint i = 0; i < SAMPLES; ++i)
  {
    vec3 dir = normalize(point.normal + RandomUnitVector(seed));
    newIrradiance += textureLod(cubemap, dir, 0.0f).rgb * max(dot(point.normal, dir), 0.0f);
  }

  newIrradiance = (prevIrradiance * float(params.temporalCount) + newIrradiance) / float(params.temporalCount + 1);
  out_temporalDiffuseIrradiance = vec4(newIrradiance, 1.0f);

  vec3 L0 = newIrradiance / float(SAMPLES);
  L0 *= LambertianDiffuseBRDF(point);

  out_color = vec4(L0, 1.0f);
}
