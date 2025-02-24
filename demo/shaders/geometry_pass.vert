#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.glsl"

#include "CameraData.h"

//==================================================================================================
// Descriptor bindings / push constants
//--------------------------------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform prev_camera_data_t
{
  CameraData prevCamera;
};

layout(set = 0, binding = 1) uniform curr_camera_data_t
{
  CameraData currCamera;
};

layout(push_constant) uniform params_t
{
  mat4 prevModel;
  mat4 currModel;
  mat3 normalMatrix;

  vec3 albedo;
  float metalness;
  float roughness;
} params;
//==================================================================================================

//==================================================================================================
// Stage linkage
//--------------------------------------------------------------------------------------------------
layout(location = 0) in vec4 posNorm;
layout(location = 1) in vec4 texCoordAndTang;

layout(location = 0) out vs_out_t
{
  vec3 wsPos;
  vec3 wsNorm;
  vec2 texCoord;

  vec4 prevPosClip;
  vec4 currPosClip;
} out_vertex;

out gl_PerVertex { vec4 gl_Position; };
//==================================================================================================

void main(void)
{
  const vec4 wNorm       = vec4(decode_normal(floatBitsToInt(posNorm.w)),         0.0f);
  const vec4 wTang       = vec4(decode_normal(floatBitsToInt(texCoordAndTang.z)), 0.0f);

  out_vertex.wsPos       = (params.currModel * vec4(posNorm.xyz, 1.0f)).xyz;
  out_vertex.wsNorm      = normalize(params.normalMatrix * wNorm.xyz);
  out_vertex.texCoord    = texCoordAndTang.xy;

  out_vertex.prevPosClip = prevCamera.projView * params.prevModel * vec4(posNorm.xyz, 1.0f);
  out_vertex.currPosClip = currCamera.projView * vec4(out_vertex.wsPos, 1.0f);

  gl_Position            = out_vertex.currPosClip;
}
