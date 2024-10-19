#ifndef NORMAL_PERTURBATION_GLSL_INCLUDED
#define NORMAL_PERTURBATION_GLSL_INCLUDED

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
vec3 PerturbNormal(sampler2D texNorm, vec3 wsNorm, vec3 wsPos, vec2 texCoord)
{
  vec3 map = 255.0f / 127.0f * texture(texNorm, texCoord).xyz - 128.0f / 127.0f;
  mat3 tbn = ConstructCotangentFrame(wsNorm, wsPos, texCoord);
  return normalize(tbn * map);
}

#endif // NORMAL_PERTURBATION_GLSL_INCLUDED
