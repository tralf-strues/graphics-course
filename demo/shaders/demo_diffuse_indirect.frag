#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "CameraData.h"
#include "NormalPerturbation.glsl"
#include "PBR.glsl"

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
//==================================================================================================

void main()
{
  const float DELTA        = 0.35f;
  const vec2  SAMPLE_COUNT = vec2(TWO_PI / DELTA, HALF_PI / DELTA);

  vec3 emissive = texture(texEmissive, vertex.texCoord).rgb;

  SurfacePoint point;
  point.normal    = PerturbNormal(texNorm, normalize(vertex.wsNorm), vertex.wsPos, vertex.texCoord);
  point.position  = vertex.wsPos;
  // point.toCam     = normalize(camera.wsPos - point.position);
  point.albedo    = texture(texAlbedo, vertex.texCoord).rgb;
  // point.metalness = texture(texMetalnessRoughness, vertex.texCoord).b;
  // point.roughness = texture(texMetalnessRoughness, vertex.texCoord).g;
  // point.f0        = vec3(0.04f);
  // point.f0        = mix(point.f0, point.albedo, point.metalness);

  vec3 brdf       = LambertianDiffuseBRDF(point);

  vec3 tang       = vec3(1.0f, 0.0f, 0.0f);
  vec3 bitang     = normalize(cross(point.normal, tang));
  tang            = normalize(cross(bitang, point.normal));

  vec3 L0 = vec3(0.0f);
  for (float angle1 = 0.0f; angle1 < TWO_PI; angle1 += DELTA)
  {
    for (float angle2 = 0.0f; angle2 < HALF_PI; angle2 += DELTA)
    {
      vec3 dir = vec3(sin(angle2) * cos(angle1), sin(angle2) * sin(angle1), cos(angle2));
      dir = mat3(tang, bitang, point.normal) * dir;

      L0 += texture(cubemap, dir).rgb * max(dot(point.normal, dir), 0.0f) * sin(angle2);
    }
  }

  L0 *= brdf;
  L0 /= (SAMPLE_COUNT.x * SAMPLE_COUNT.y);

  out_color = vec4(L0, 1.0f);
}
