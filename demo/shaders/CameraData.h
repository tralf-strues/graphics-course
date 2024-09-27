#ifndef CAMERA_DATA_H_INCLUDED
#define CAMERA_DATA_H_INCLUDED

#include "cpp_glsl_compat.h"

struct CameraData
{
  shader_mat4 projView;
  shader_mat4 view;

  shader_vec3 wsPos;
  shader_vec3 wsRight;
  shader_vec3 wsUp;
  shader_vec3 wsForward;
};

#endif // CAMERA_DATA_H_INCLUDED
