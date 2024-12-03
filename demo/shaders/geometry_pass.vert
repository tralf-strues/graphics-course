#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.glsl"

#include "CameraData.h"

//==================================================================================================
// Descriptor bindings / push constants
//--------------------------------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform camera_data_t
{
  CameraData camera;
};

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
layout(location = 0) in vec4 posNorm;
layout(location = 1) in vec4 texCoordAndTang;

layout(location = 0) out vs_out_t
{
  vec3 wsPos;
  vec3 wsNorm;
  vec2 texCoord;
} out_vertex;

out gl_PerVertex { vec4 gl_Position; };
//==================================================================================================

void main(void)
{
  const vec4 wNorm     = vec4(decode_normal(floatBitsToInt(posNorm.w)),         0.0f);
  const vec4 wTang     = vec4(decode_normal(floatBitsToInt(texCoordAndTang.z)), 0.0f);

  out_vertex.wsPos     = (params.model * vec4(posNorm.xyz, 1.0f)).xyz;
  out_vertex.wsNorm    = normalize(params.normalMatrix * wNorm.xyz);
  out_vertex.texCoord  = texCoordAndTang.xy;

  gl_Position          = camera.projView * vec4(out_vertex.wsPos, 1.0f);
}
