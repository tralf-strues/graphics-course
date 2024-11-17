#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "CameraData.h"
#include "NormalPerturbation.glsl"
#include "PBR.glsl"
#include "Random.glsl"

const float A0 = 3.141593;
const float A1 = 2.094395;
const float A2 = 0.785398;

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

layout(set = 1, binding = 4, std430) readonly buffer Coeffs {
    float sphericalHarmonicCoeffs[27];
};

layout(push_constant) uniform params_t
{
  mat4 model;
  mat3 normalMatrix;
  uint temporalCount;
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

  float Y00     = 0.282095;
  float Y11     = 0.488603 * point.normal.x;
  float Y10     = 0.488603 * point.normal.z;
  float Y1_1    = 0.488603 * point.normal.y;
  float Y21     = 1.092548 * point.normal.x * point.normal.z;
  float Y2_1    = 1.092548 * point.normal.y * point.normal.z;
  float Y2_2    = 1.092548 * point.normal.y * point.normal.x;
  float Y20     = 0.946176 * point.normal.z * point.normal.z - 0.315392;
  float Y22     = 0.546274 * (point.normal.x * point.normal.x - point.normal.y * point.normal.y);

  vec3 L00   = vec3(sphericalHarmonicCoeffs[0 ], sphericalHarmonicCoeffs[0  + 1], sphericalHarmonicCoeffs[0  + 2]);
  vec3 L11   = vec3(sphericalHarmonicCoeffs[3 ], sphericalHarmonicCoeffs[3  + 1], sphericalHarmonicCoeffs[3  + 2]);
  vec3 L10   = vec3(sphericalHarmonicCoeffs[6 ], sphericalHarmonicCoeffs[6  + 1], sphericalHarmonicCoeffs[6  + 2]);
  vec3 L1_1  = vec3(sphericalHarmonicCoeffs[9 ], sphericalHarmonicCoeffs[9  + 1], sphericalHarmonicCoeffs[9  + 2]);
  vec3 L21   = vec3(sphericalHarmonicCoeffs[12], sphericalHarmonicCoeffs[12 + 1], sphericalHarmonicCoeffs[12 + 2]);
  vec3 L2_1  = vec3(sphericalHarmonicCoeffs[15], sphericalHarmonicCoeffs[15 + 1], sphericalHarmonicCoeffs[15 + 2]);
  vec3 L2_2  = vec3(sphericalHarmonicCoeffs[18], sphericalHarmonicCoeffs[18 + 1], sphericalHarmonicCoeffs[18 + 2]);
  vec3 L20   = vec3(sphericalHarmonicCoeffs[21], sphericalHarmonicCoeffs[21 + 1], sphericalHarmonicCoeffs[21 + 2]);
  vec3 L22   = vec3(sphericalHarmonicCoeffs[24], sphericalHarmonicCoeffs[24 + 1], sphericalHarmonicCoeffs[24 + 2]);

  vec3 E = A0*Y00*L00
             + A1*Y1_1*L1_1 + A1*Y10*L10 + A1*Y11*L11 
             + A2*Y2_2*L2_2 + A2*Y2_1*L2_1 + A2*Y20*L20 + A2*Y21*L21 + A2*Y22*L22;

  vec3 L0 = E * LambertianDiffuseBRDF(point);

  out_color = vec4(L0, 1.0f);
}
