#ifndef LIGHT_H_INCLUDED
#define LIGHT_H_INCLUDED

#include "cpp_glsl_compat.h"

struct DirectionalLight
{
  shader_vec3 color;
  float intensity;

  shader_vec3 direction;
  float _pad0;
};

struct PointLight
{
  shader_vec3 color;
  float intensity;

  shader_vec3 position;
  float radius;
};

#endif // LIGHT_H_INCLUDED
