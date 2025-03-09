#ifndef CAMERA_DATA_H_INCLUDED
#define CAMERA_DATA_H_INCLUDED

#include "cpp_glsl_compat.h"

struct CameraData
{
  shader_mat4 projView;
  shader_mat4 proj;
  shader_mat4 view;

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

#endif // CAMERA_DATA_H_INCLUDED
