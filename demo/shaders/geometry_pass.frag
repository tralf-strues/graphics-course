#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

//==================================================================================================
// Descriptor bindings / push constants
//--------------------------------------------------------------------------------------------------
layout(set = 1, binding = 0) uniform sampler2D texAlbedo;
layout(set = 1, binding = 1) uniform sampler2D texMetalnessRoughness;
layout(set = 1, binding = 2) uniform sampler2D texNorm;
//--------------------------------------------------------------------------------------------------

//==================================================================================================
// Stage linkage
//--------------------------------------------------------------------------------------------------
layout(location = 0) in vs_out_t
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
} vertex;

// R8G8B8A8_UNORM
// - R8G8B8: albedo
// - A8:     reserved for future use
layout(location = 0) out vec4 out_albedo;

// R8G8B8A8_UNORM
// - R8: metalness
// - G8: roughness
// - B8: reserved for future use
// - A8: reserved for future use
layout(location = 1) out vec4 out_metalnessRoughness;

// R10G10B10A2
// - R10G10B10: world-space normal
// - A2:        reserved for future use
layout(location = 2) out vec4 out_wNorm;
//==================================================================================================

void main()
{
  /* Albedo */
  out_albedo = vec4(texture(texAlbedo, vertex.texCoord).rgb, 1.0f);

  /* Metalness & Roughness */
  out_metalnessRoughness = vec4(texture(texMetalnessRoughness, vertex.texCoord).rg, 0.0f, 0.0f);

  /* Normal */
  const vec3 t   = normalize(vertex.wTangent);
  const vec3 n   = normalize(vertex.wNorm);
  const vec3 b   = normalize(cross(n, t));
  const mat3 tbn = mat3(t, b, n);

  vec3 normal    = 2.0f * texture(texNorm, vertex.texCoord).xyz - 1.0f;
  normal         = normalize(tbn * normal);

  out_wNorm      = vec4(0.5f * normal + 0.5f, 0.0f);
}
