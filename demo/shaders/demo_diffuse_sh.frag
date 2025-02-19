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

layout(set = 1, binding = 4, scalar) readonly buffer Coeffs {
  vec3 E_lm[9];
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
  point.albedo    = texture(texAlbedo, vertex.texCoord).rgb;

  float Y_lm[9] = {
    0.282095,
    0.488603 * point.normal.x,
    0.488603 * point.normal.z,
    0.488603 * point.normal.y,
    1.092548 * point.normal.x * point.normal.z,
    1.092548 * point.normal.y * point.normal.z,
    1.092548 * point.normal.y * point.normal.x,
    0.946176 * point.normal.z * point.normal.z - 0.315392,
    0.546274 * (point.normal.x * point.normal.x - point.normal.y * point.normal.y),
  };

  vec3 E = vec3(0.0f);
  for (uint i = 0; i < 9; ++i)
  {
    E += E_lm[i] * Y_lm[i];
  }

  vec3 L0 = E * LambertianDiffuseBRDF(point);

  out_color = vec4(L0, 1.0f);
}
