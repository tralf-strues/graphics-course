#ifndef LIGHT_H_INCLUDED
#define LIGHT_H_INCLUDED

#include "cpp_glsl_compat.h"

struct DirectionalLight
{
  shader_vec3 radiance;
  float _pad0;

  shader_vec3 direction;
  float _pad1;
};

struct PointLight
{
  shader_vec3 radiance;
  float radius;

  shader_vec3 position;
  float _pad0;
};

#endif // LIGHT_H_INCLUDED
