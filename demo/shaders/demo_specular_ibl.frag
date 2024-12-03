#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : enable

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

layout(set = 1, binding = 4) uniform samplerCube texPrefilteredEnvMap;
layout(set = 1, binding = 5) uniform sampler2D texEnvBRDF;

layout(push_constant) uniform params_t
{
  mat4 model;
  mat3 normalMatrix;
  uint temporalCount;
  float metalness;
  float roughness;
  uint envMapMips;
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

vec3 SpecularIBL(SurfacePoint point) {
  float NoV = clamp(dot(point.normal, point.toCam), 0.5f / 512.0f, 1.0f);
  vec3 R = reflect(-point.toCam, point.normal);

  // Some remapping found in moving frostbite to PBR,
  // yet to be understood :)
  float smoothness = clamp(1.0f - point.roughness, 0.0f, 1.0f);
  float lerpFactor = smoothness * (sqrt(smoothness) + point.roughness);
  R = mix(point.normal, R, lerpFactor);

  float mip = point.roughness * float(params.envMapMips - 1);
  vec3 prefilteredColor = textureLod(texPrefilteredEnvMap, R, mip).rgb;

  vec2 envBrdf = texture(texEnvBRDF, vec2(NoV, point.roughness)).rg;

  vec3 F = FresnelSchlick(point.normal, point.toCam, point.f0);

  // return vec3(envBrdf.r, envBrdf.g, 0.0f);
  return prefilteredColor * (F * envBrdf.r + envBrdf.g);
}

void main()
{
  SurfacePoint point;
  point.normal    = PerturbNormal(texNorm, normalize(vertex.wsNorm), vertex.wsPos, vertex.texCoord);
  point.position  = vertex.wsPos;
  point.toCam     = normalize(camera.wsPos - point.position);
  point.albedo    = texture(texAlbedo, vertex.texCoord).rgb;

  point.metalness = params.metalness;
  point.roughness = texture(texMetalnessRoughness, vertex.texCoord).g * params.roughness;

  point.f0        = vec3(0.04f);
  point.f0        = mix(point.f0, point.albedo, point.metalness);

  vec3 L0         = SpecularIBL(point);
  out_color       = vec4(L0, 1.0f);
}
