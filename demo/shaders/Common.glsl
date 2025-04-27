#ifndef COMMON_GLSL_INCLUDED
#define COMMON_GLSL_INCLUDED

const float PI         = 3.14159265359f;
const float TWO_PI     = 2.0f * PI;
const float HALF_PI    = 0.5f * PI;
const float INV_PI     = 1.0f / PI;
const float INV_TWO_PI = 1.0f / TWO_PI;

vec3 LoadNormal(sampler2D texNorm, vec2 uv)
{
  vec3 normalWS = texture(texNorm, uv).xyz;
  normalWS = normalize(255.0f / 127.0f * normalWS - 128.0f / 127.0f);

  return normalWS;
}

#endif // COMMON_GLSL_INCLUDED
