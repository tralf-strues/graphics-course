#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "NormalPerturbation.glsl"
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

layout(set = 1, binding = 0) uniform sampler2D texAlbedo;
layout(set = 1, binding = 1) uniform sampler2D texMetalnessRoughness;
layout(set = 1, binding = 2) uniform sampler2D texNorm;
layout(set = 1, binding = 3) uniform sampler2D texEmissive;

layout(push_constant) uniform params_t
{
  mat4 prevModel;
  mat4 currModel;
  mat3 normalMatrix;

  vec3 albedo;
  float metalness;
  float roughness;
  bool unjitterTextureUVs;
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

  vec4 prevPosClip;
  vec4 currPosClip;
} vertex;

// R8G8B8A8_UNORM
// - R8G8B8: albedo
// - A8:     emissive r channel
layout(location = 0) out vec4 out_albedoEmissiveR;

// R8G8B8A8_UNORM
// - R8: metalness
// - G8: roughness
// - B8: emissive g channel
// - A8: emissive b channel
layout(location = 1) out vec4 out_metalnessRoughnessEmissiveGB;

// R10G10B10A2
// - R10G10B10: world-space normal
// - A2:        reserved for future use
layout(location = 2) out vec4 out_wsNorm;

// R16G16
layout(location = 3) out vec2 out_motionVectors;
//==================================================================================================

vec2 UnjitterTextureUV(vec2 uv)
{
  if (params.unjitterTextureUVs)
  {
    return uv - dFdx(uv) * currCamera.jitterPixels.x + dFdy(uv) * currCamera.jitterPixels.y;
  }

  return uv;
}

void main()
{
  vec2 texCoord = UnjitterTextureUV(vertex.texCoord);

  vec3 emissive = texture(texEmissive, texCoord).rgb;

  /* Albedo */
  out_albedoEmissiveR = vec4(texture(texAlbedo, texCoord).rgb, emissive.r);
  out_albedoEmissiveR.rgb *= params.albedo;

  /* Metalness & Roughness */
  out_metalnessRoughnessEmissiveGB =
    vec4(texture(texMetalnessRoughness, texCoord).bg, emissive.gb);
  out_metalnessRoughnessEmissiveGB.xy *= vec2(params.metalness, params.roughness);

  /* Normal */
  vec3 norm = PerturbNormal(texNorm, normalize(vertex.wsNorm), vertex.wsPos, texCoord);
  out_wsNorm = vec4(0.5f * norm + 0.5f, 0.0f);

  /* Motion Vectors */
  vec2 prevPosNDC = vertex.prevPosClip.xy / vertex.prevPosClip.w; // Perspective divide
  vec2 currPosNDC = vertex.currPosClip.xy / vertex.currPosClip.w;

  vec2 motionVectorUV = 0.5f * (prevPosNDC - currPosNDC);
  motionVectorUV -= 0.5f * prevCamera.jitterNDC;
  motionVectorUV += 0.5f * currCamera.jitterNDC;

  out_motionVectors = motionVectorUV;
}
