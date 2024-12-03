#ifndef RANDOM_GLSL_INCLUDED
#define RANDOM_GLSL_INCLUDED

#include "Common.glsl"

// Reference: https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
uint WangHash(inout uint seed) {
  seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
  seed *= uint(9);
  seed = seed ^ (seed >> 4);
  seed *= uint(0x27d4eb2d);
  seed = seed ^ (seed >> 15);

  return seed;
}

float RandomFloatNormalized(inout uint seed) {
  return float(WangHash(seed)) / 4294967296.0f;
}

vec3 RandomUnitVector(inout uint seed) {
  float z = RandomFloatNormalized(seed) * 2.0f - 1.0f;
  float a = RandomFloatNormalized(seed) * TWO_PI;
  float r = sqrt(1.0f - z * z);
  float x = r * cos(a);
  float y = r * sin(a);

  return vec3(x, y, z);
}

vec2 Hammersley(uint i, uint samples) {
  uint bits = i;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);

  float uniformX = float(bits) * 2.3283064365386963e-10f;
  float uniformY = float(i) / float(samples);

  return vec2(uniformX, uniformY);
}

#endif // RANDOM_GLSL_INCLUDED
