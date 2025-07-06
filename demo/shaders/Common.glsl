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

float Luminance(vec3 color)
{
  return max(dot(color, vec3(0.299f, 0.587f, 0.114f)), 0.001f);
}

vec3 HistoryClampAABB(vec3 aabbMin, vec3 aabbMax, vec3 history)
{
  return clamp(history, aabbMin, aabbMax);
}

vec3 HistoryClipAABB(vec3 aabbMin, vec3 aabbMax, vec3 history)
{
    vec3 center = 0.5f * (aabbMax + aabbMin);
    vec3 clipRadius = 0.5f * (aabbMax - aabbMin) + 0.001f;

    vec3 toHistory = history - center;
    vec3 toHistoryClipped = toHistory / clipRadius;
    toHistoryClipped = abs(toHistoryClipped);

    float maxAxis = max(max(toHistoryClipped.x, toHistoryClipped.y), toHistoryClipped.z);
    return (maxAxis > 1.0f) ? (center + toHistory / maxAxis) : history;
}

#endif // COMMON_GLSL_INCLUDED
