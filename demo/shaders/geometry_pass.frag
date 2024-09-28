#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

//==================================================================================================
// Descriptor bindings / push constants
//--------------------------------------------------------------------------------------------------
layout(set = 1, binding = 0) uniform sampler2D texAlbedo;
layout(set = 1, binding = 1) uniform sampler2D texMetalnessRoughness;
layout(set = 1, binding = 2) uniform sampler2D texNorm;
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
layout(location = 2) out vec4 out_wsNorm;
//==================================================================================================

mat3 ConstructCotangentFrame(vec3 wsNorm, vec3 wsPos, vec2 texCoord)
{
  vec3 dp1      = dFdx(wsPos);
  vec3 dp2      = dFdy(wsPos);
  vec2 duv1     = dFdx(texCoord);
  vec2 duv2     = dFdy(texCoord);

  vec3 dp1perp  = cross(wsNorm, dp1);
  vec3 dp2perp  = cross(dp2, wsNorm);

  vec3 wsTang   = dp2perp * duv1.x + dp1perp * duv2.x;
  vec3 wsBitang = dp2perp * duv1.y + dp1perp * duv2.y;
  float invmax  = inversesqrt(max(dot(wsTang, wsTang), dot(wsBitang, wsBitang)));

  return mat3(wsTang * invmax, wsBitang * invmax, wsNorm);
}

// Normal perturbation without precomputed tangents.
// Borrowed from: http://www.thetenthplanet.de/archives/1180
vec3 PerturbNormal(vec3 wsNorm, vec3 wsPos, vec2 texCoord)
{
  vec3 map = 255.0f / 127.0f * texture(texNorm, vertex.texCoord).xyz - 128.0f / 127.0f;
  mat3 tbn = ConstructCotangentFrame(wsNorm, wsPos, texCoord);
  return normalize(tbn * map);
}

void main()
{
  /* Albedo */
  out_albedo = vec4(texture(texAlbedo, vertex.texCoord).rgb, 1.0f);

  /* Metalness & Roughness */
  out_metalnessRoughness = vec4(texture(texMetalnessRoughness, vertex.texCoord).bg, 0.0f, 0.0f);

  /* Normal */
  out_wsNorm = vec4(PerturbNormal(vertex.wsNorm, vertex.wsPos, vertex.texCoord), 0.0f);
}
