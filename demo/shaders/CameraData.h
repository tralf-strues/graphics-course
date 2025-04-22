#ifndef CAMERA_DATA_H_INCLUDED
#define CAMERA_DATA_H_INCLUDED

#include "cpp_glsl_compat.h"

struct CameraData
{
  shader_mat4 projView;
  shader_mat4 proj;
  shader_mat4 view;

  float proj22;
  float proj23;
  float invProj00; // Precomputed coefficient, 1 / proj[0][0]
  float invProj11; // Precomputed coefficient, 1 / proj[1][1]

  shader_vec3 wsPos;
  float _pad0;

  shader_vec3 wsRight;
  float _pad1;

  shader_vec3 wsUp;
  float _pad2;

  shader_vec3 wsForward;
  float _pad3;

  shader_vec2 jitterNDC;
  shader_vec2 jitterPixels;
};

#ifndef __cplusplus

vec3 ConvertWorldToScreen(CameraData camera, vec3 posWS)
{
  vec4 csPos = camera.projView * vec4(posWS, 1.0f);
  csPos /= csPos.w;

  vec3 ssPos = csPos.xyz;
  ssPos.xy = 0.5f * ssPos.xy + 0.5f;

  return ssPos;
}

vec3 ConvertScreenToWorld(CameraData camera, vec3 posSS)
{
  // Position reconstruction from depth.
  // Inspired by: https://mynameismjp.wordpress.com/2010/09/05/position-from-depth-3/
  vec2  ndcXY   = 2.0f * posSS.xy - 1.0f;
  float depthVS = camera.proj23 / (posSS.z - camera.proj22);
  vec3  posWS   = camera.wsPos + depthVS * (camera.wsForward -
                                            camera.wsRight * ndcXY.x * camera.invProj00 +
                                            camera.wsUp    * ndcXY.y * camera.invProj11);

  return posWS;
}

vec3 ConvertDirectionWorldToScreen(CameraData camera, vec3 fromWS, vec3 fromSS, vec3 directionWS)
{
  vec3 directionEndWS = fromWS + directionWS;
  vec3 directionEndSS = ConvertWorldToScreen(camera, directionEndWS);
  return normalize(directionEndSS - fromSS);
}

#endif // __cplusplus

#endif // CAMERA_DATA_H_INCLUDED
