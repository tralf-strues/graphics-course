#ifndef CAMERA_DATA_H_INCLUDED
#define CAMERA_DATA_H_INCLUDED

#include "cpp_glsl_compat.h"

struct CameraData
{
  shader_mat4 projView;
  shader_mat4 view;
  shader_vec4 wPos;
};

#endif // CAMERA_DATA_H_INCLUDED
