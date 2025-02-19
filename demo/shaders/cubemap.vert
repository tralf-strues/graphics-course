#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "CameraData.h"

//==================================================================================================
// Descriptor bindings / push constants
//--------------------------------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform camera_data_t
{
  CameraData camera;
};
//==================================================================================================

//==================================================================================================
// Stage linkage
//--------------------------------------------------------------------------------------------------
layout(location = 0) out vs_out_t
{
  vec3 pos;
} out_vertex;

out gl_PerVertex { vec4 gl_Position; };
//==================================================================================================

vec3 CUBE_VERTEX_POSITIONS[] = {
  // back face
  vec3(-1.0f, -1.0f, -1.0f), // bottom-left
  vec3(1.0f,  1.0f, -1.0f), // top-right
  vec3(1.0f, -1.0f, -1.0f), // bottom-right
  vec3(1.0f,  1.0f, -1.0f), // top-right
  vec3(-1.0f, -1.0f, -1.0f), // bottom-left
  vec3(-1.0f,  1.0f, -1.0f), // top-left
  // front face
  vec3(-1.0f, -1.0f,  1.0f), // bottom-left
  vec3(1.0f, -1.0f,  1.0f), // bottom-right
  vec3(1.0f,  1.0f,  1.0f), // top-right
  vec3(1.0f,  1.0f,  1.0f), // top-right
  vec3(-1.0f,  1.0f,  1.0f), // top-left
  vec3(-1.0f, -1.0f,  1.0f), // bottom-left
  // left face
  vec3(-1.0f,  1.0f,  1.0f), // top-right
  vec3(-1.0f,  1.0f, -1.0f), // top-left
  vec3(-1.0f, -1.0f, -1.0f), // bottom-left
  vec3(-1.0f, -1.0f, -1.0f), // bottom-left
  vec3(-1.0f, -1.0f,  1.0f), // bottom-right
  vec3(-1.0f,  1.0f,  1.0f), // top-right
  // right face
  vec3(1.0f,  1.0f,  1.0f), // top-left
  vec3(1.0f, -1.0f, -1.0f), // bottom-right
  vec3(1.0f,  1.0f, -1.0f), // top-right
  vec3(1.0f, -1.0f, -1.0f), // bottom-right
  vec3(1.0f,  1.0f,  1.0f), // top-left
  vec3(1.0f, -1.0f,  1.0f), // bottom-left
  // bottom face
  vec3(-1.0f, -1.0f, -1.0f), // top-right
  vec3(1.0f, -1.0f, -1.0f), // top-left
  vec3(1.0f, -1.0f,  1.0f), // bottom-left
  vec3(1.0f, -1.0f,  1.0f), // bottom-left
  vec3(-1.0f, -1.0f,  1.0f), // bottom-right
  vec3(-1.0f, -1.0f, -1.0f), // top-right
  // top face
  vec3(-1.0f,  1.0f, -1.0f), // top-left
  vec3(1.0f,  1.0f , 1.0f), // bottom-right
  vec3(1.0f,  1.0f, -1.0f), // top-right
  vec3(1.0f,  1.0f,  1.0f), // bottom-right
  vec3(-1.0f,  1.0f, -1.0f), // top-left
  vec3(-1.0f,  1.0f,  1.0f), // bottom-left
};

void main(void)
{
  out_vertex.pos = CUBE_VERTEX_POSITIONS[gl_VertexIndex];

  mat4 view = mat4(mat3(camera.view));
  gl_Position = (camera.proj * view * vec4(out_vertex.pos, 1.0f)).xyww;
}
